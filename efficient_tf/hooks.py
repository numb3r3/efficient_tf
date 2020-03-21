import os
import glob
import shutil
import json

import tensorflow as tf


class ExamplesPerSecondHook(tf.estimator.SessionRunHook):
    """Hook to print out examples per second."""

    def __init__(self, batch_size, every_n_iter=100, every_n_secs=None):
        """Initializer for ExamplesPerSecondHook."""
        if (every_n_iter is None) == (every_n_secs is None):
            raise ValueError("exactly one of every_n_steps"
                             " and every_n_secs should be provided.")
        self._timer = tf.estimator.SecondOrStepTimer(
            every_steps=every_n_iter,
            every_secs=every_n_secs)

        self._step_train_time = 0
        self._total_steps = 0
        self._batch_size = batch_size

    def begin(self):
        self._global_step_tensor = tf.train.get_global_step()
        if self._global_step_tensor is None:
            raise RuntimeError(
                "Global step should be created to use StepCounterHook.")

    def before_run(self, run_context):
        del run_context
        return tf.estimator.SessionRunArgs(self._global_step_tensor)

    def after_run(self, run_context, run_values):
        del run_context

        global_step = run_values.results
        if self._timer.should_trigger_for_step(global_step):
            elapsed_time, elapsed_steps = self._timer.update_last_triggered_step(
                global_step)
            if elapsed_time is not None:
                steps_per_sec = elapsed_steps / elapsed_time
                self._step_train_time += elapsed_time
                self._total_steps += elapsed_steps

                average_examples_per_sec = self._batch_size * (
                    self._total_steps / self._step_train_time)
                current_examples_per_sec = steps_per_sec * self._batch_size
                tf.logging.info("Examples/sec: %g (%g), step = %g",
                                          average_examples_per_sec,
                                          current_examples_per_sec,
                                          self._total_steps)
                # print("Examples/sec: %g (%g), step = %g" %
                #       (average_examples_per_sec, current_examples_per_sec,
                #        self._total_steps))


class LoggingTensorHook(tf.estimator.SessionRunHook):
    """Hook to print batch of tensors."""

    def __init__(self, collection,
                 every_n_iter=None,
                 every_n_secs=None,
                 batch=False,
                 first_k=None):
        """Initializes a `LoggingTensorHook`."""
        self._collection = collection
        self._batch = batch
        self._first_k = first_k
        self._timer = tf.estimator.SecondOrStepTimer(
            every_secs=every_n_secs,
            every_steps=every_n_iter)

    def begin(self):
        self._timer.reset()
        self._iter_count = 0

    def before_run(self, run_context):    # pylint: disable=unused-argument
        self._should_trigger = self._timer.should_trigger_for_step(
            self._iter_count)
        if self._should_trigger:
            tensors = {
                t.name: t
                for t in tf.get_collection(self._collection)
            }
            return tf.estimator.SessionRunArgs(tensors)
        else:
            return None

    def _log_tensors(self, tensor_values):
        elapsed_secs, _ = self._timer.update_last_triggered_step(
            self._iter_count)
        if self._batch:
            self._batch_print(tensor_values)
        else:
            self._print(tensor_values)

    def after_run(self, run_context, run_values):
        _ = run_context
        if self._should_trigger:
            self._log_tensors(run_values.results)

        self._iter_count += 1

    def _print(self, tensor_values):
        if not tensor_values:
            return
        for k, v in tensor_values.items():
            tf.logging.info("{0}: {1}".format(k, v))
            print("{0}: {1}".format(k, v))

    def _batch_print(self, tensor_values):
        if not tensor_values:
            return
        batch_size = list(tensor_values.values())[0].shape[0]
        if self._first_k is not None:
            batch_size = min(self._first_k, batch_size)
        for i in range(batch_size):
            for k, v in tensor_values.items():
                tf.logging.info("{0}: {1}".format(k, v[i]))
                print("{0}: {1}".format(k, v[i]))


class SummarySaverHook(tf.estimator.SessionRunHook):
    """Saves summaries every N steps."""

    def __init__(self,
                 every_n_iter=None,
                 every_n_secs=None,
                 output_dir=None,
                 summary_writer=None):
        """Initializes a `SummarySaverHook`."""
        self._summary_writer = summary_writer
        self._output_dir = output_dir
        self._timer = tf.estimator.SecondOrStepTimer(
            every_secs=every_n_iter,
            every_steps=every_n_secs)

    def begin(self):
        if self._summary_writer is None and self._output_dir:
            self._summary_writer = tf.summary.FileWriterCache.get(
                self._output_dir)
        self._next_step = None
        self._global_step_tensor = tf.train.get_global_step()
        self._summaries = tf.summary.merge_all()
        if self._global_step_tensor is None:
            raise RuntimeError(
                "Global step should be created to use SummarySaverHook.")

    def before_run(self, run_context):    # pylint: disable=unused-argument
        self._request_summary = (
            self._next_step is None or
            self._timer.should_trigger_for_step(self._next_step))
        requests = {"global_step": self._global_step_tensor}
        if self._request_summary and self._summaries is not None:
            requests["summary"] = self._summaries

        return tf.estimator.SessionRunArgs(requests)

    def after_run(self, run_context, run_values):
        _ = run_context
        if not self._summary_writer:
            return

        global_step = run_values.results["global_step"]

        if self._request_summary:
            self._timer.update_last_triggered_step(global_step)
            if "summary" in run_values.results:
                summary = run_values.results["summary"]
                self._summary_writer.add_summary(summary, global_step)

        self._next_step = global_step + 1

    def end(self, session=None):
        if self._summary_writer:
            self._summary_writer.flush()


class CallAfterSaveListener(tf.estimator.CheckpointSaverListener):

    def __init__(self, callback, delay_steps=10):
        self._callback = callback
        self._delay_steps = delay_steps

    def after_save(self, session, global_step_value):
        if global_step_value > self._delay_steps:
            self._callback()
        return False


def serialize_json(data, indent=4, sort_keys=True):
    import ast

    try:
        data = ast.literal_eval(str(data))
    except:
        data = json.loads(str(data))

    return json.dumps(data, indent=indent, sort_keys=sort_keys)


class BestCheckpointKeeper(tf.estimator.CheckpointSaverListener):

    def __init__(self, model_dir, eval_fn,
                 eval_set="eval",
                 eval_metric="loss",
                 compare_fn="less",
                 delay_steps=10):
        self._model_dir = model_dir
        self._eval_fn = eval_fn
        self._delay_steps = delay_steps
        self._eval_set = eval_set
        self._eval_metric = eval_metric
        self._compare_fn = compare_fn
        self._best_eval_result = {}

    def after_save(self, session, global_step_value):
        if global_step_value > self._delay_steps:
            self._eval_and_save()
        return False

    def _serialize_json(self, data, indent=4, sort_keys=True):
        import ast

        try:
            data = ast.literal_eval(str(data))
        except:
            data = json.loads(str(data))

        return json.dumps(data, indent=indent, sort_keys=sort_keys)

    def _eval_and_save(self):
        results = self._eval_fn()
        if self._result_is_best(results):
            output_dir = os.path.join(self._model_dir, 'best')
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            # clean old files
            for root, dirs, files in os.walk(output_dir):
                for f in files:
                    os.unlink(os.path.join(root, f))
                for d in dirs:
                    shutil.rmtree(os.path.join(root, d))

            checkpoint_path = tf.train.latest_checkpoint(self._model_dir)
            print("Saving the new best checkpoint:", checkpoint_path)
            for path in glob.glob(checkpoint_path + "*"):
                shutil.copy(path, output_dir)
            results_path = os.path.join(
                output_dir,
                os.path.basename(checkpoint_path) + "_results.json")
            with open(results_path, "w") as f:
                f.write(self._serialize_json(results))

    def _result_is_best(self, current_eval_result):
        if (not self._eval_set in current_eval_result):
            print(self._eval_set, "doesn't exist")
            return False

        if (not self._eval_metric in current_eval_result[self._eval_set]):
            print(self._eval_metric, "doesn't exist")
            return False

        if not self._best_eval_result:
            self._best_eval_result = current_eval_result
            return True

        current_val = current_eval_result[self._eval_set][self._eval_metric]
        best_val = self._best_eval_result[self._eval_set][self._eval_metric]

        pred = {
            "less": current_val < best_val,
            "greater": current_val > best_val
        }[self._compare_fn]

        if pred:
            self._best_eval_result = current_eval_result

        return pred
