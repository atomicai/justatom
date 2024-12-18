import logging
import numbers

import numpy as np
import torch
from tqdm import tqdm

from farm.evaluation.metrics import compute_metrics, compute_report_metrics
from farm.utils import WANDBLogger as MlLogger
from farm.utils import to_numpy
from farm.visual.ascii.images import BUSH_SEP

logger = logging.getLogger(__name__)


class Evaluator:
    """Handles evaluation of a given model over a specified dataset."""

    def __init__(self, data_loader, tasks, device, report=True):
        """
        :param data_loader: The PyTorch DataLoader that will return batches of data from the evaluation dataset
        :type data_loader: DataLoader
        :param label_maps:
        :param device: The device on which the tensors should be processed. Choose from "cpu" and "cuda".
        :param metrics: The list of metrics which need to be computed, one for each prediction head.
        :param metrics: list
        :param report: Whether an eval report should be generated (e.g. classification report per class).
        :type report: bool
        """

        self.data_loader = data_loader
        self.tasks = tasks
        self.device = device
        self.report = report

    def eval(self, model, return_preds_and_labels=False, calibrate_conf_scores=False):
        """
        Performs evaluation on a given model.

        :param model: The model on which to perform evaluation
        :type model: AdaptiveModel
        :param return_preds_and_labels: Whether to add preds and labels in the returned dicts of the
        :type return_preds_and_labels: bool
        :param calibrate_conf_scores: Whether to calibrate the temperature for temperature scaling of the confidence scores
        :type calibrate_conf_scores: bool
        :return all_results: A list of dictionaries, one for each prediction head. Each dictionary contains the metrics
                             and reports generated during evaluation.
        :rtype all_results: list of dicts
        """
        model.eval()

        # init empty lists per prediction head
        loss_all = [0 for _ in model.prediction_heads]
        preds_all = [[] for _ in model.prediction_heads]
        label_all = [[] for _ in model.prediction_heads]
        ids_all = [[] for _ in model.prediction_heads]
        passage_start_t_all = [[] for _ in model.prediction_heads]
        logits_all = [[] for _ in model.prediction_heads]

        for step, batch in enumerate(tqdm(self.data_loader, desc="Evaluating", mininterval=10)):  # noqa: B007
            batch = {key: batch[key].to(self.device) for key in batch}

            with torch.no_grad():
                logits = model.forward(**batch)
                losses_per_head = model.logits_to_loss_per_head(logits=logits, **batch)
                preds = model.logits_to_preds(logits=logits, **batch)
                labels = model.prepare_labels(**batch)

            # stack results of all batches per prediction head
            for head_num, head in enumerate(model.prediction_heads):
                loss_all[head_num] += np.sum(to_numpy(losses_per_head[head_num]))
                preds_all[head_num] += list(to_numpy(preds[head_num]))
                label_all[head_num] += list(to_numpy(labels[head_num]))
                if head.model_type == "span_classification":
                    ids_all[head_num] += list(to_numpy(batch["id"]))
                    passage_start_t_all[head_num] += list(to_numpy(batch["passage_start_t"]))
                    if calibrate_conf_scores:
                        logits_all[head_num] += list(to_numpy(logits))

        # Evaluate per prediction head
        all_results = []
        for head_num, head in enumerate(model.prediction_heads):
            if head.model_type == "multilabel_text_classification":
                # converting from string preds back to multi-hot encoding
                from sklearn.preprocessing import MultiLabelBinarizer

                mlb = MultiLabelBinarizer(classes=head.label_list)
                # TODO check why .fit() should be called on predictions, rather than on labels
                preds_all[head_num] = mlb.fit_transform(preds_all[head_num])
                label_all[head_num] = mlb.transform(label_all[head_num])
            if head.model_type == "span_classification" and calibrate_conf_scores:
                temperature_previous = head.temperature_for_confidence.item()
                logger.info(f"temperature used for confidence scores before calibration: {temperature_previous}")
                head.calibrate_conf(logits_all[head_num], label_all[head_num])
                temperature_current = head.temperature_for_confidence.item()
                logger.info(f"temperature used for confidence scores after calibration: {temperature_current}")
                temperature_change = (abs(temperature_current - temperature_previous) / temperature_previous) * 100.0
                if temperature_change > 50:
                    logger.warning(
                        f"temperature used for calibration of confidence scores changed by more than {temperature_change} percent"
                    )
            if hasattr(head, "aggregate_preds"):
                # Needed to convert NQ ids from np arrays to strings
                ids_all_str = [x.astype(str) for x in ids_all[head_num]]
                ids_all_list = [list(x) for x in ids_all_str]
                head_ids = ["-".join(x) for x in ids_all_list]
                preds_all[head_num], label_all[head_num] = head.aggregate_preds(
                    preds=preds_all[head_num],
                    labels=label_all[head_num],
                    passage_start_t=passage_start_t_all[head_num],
                    ids=head_ids,
                )

            result = {"loss": loss_all[head_num] / len(self.data_loader.dataset), "task_name": head.task_name}
            result.update(compute_metrics(metric=head.metric, preds=preds_all[head_num], labels=label_all[head_num]))

            # Select type of report depending on prediction head output type
            if self.report:
                try:
                    result["report"] = compute_report_metrics(head, preds_all[head_num], label_all[head_num])
                except:  # noqa: E722
                    logger.error(
                        f"Couldn't create eval report for head {head_num} with following preds and labels:"
                        f"\n Preds: {preds_all[head_num]} \n Labels: {label_all[head_num]}"
                    )
                    result["report"] = "Error"

            if return_preds_and_labels:
                result["preds"] = preds_all[head_num]
                result["labels"] = label_all[head_num]

            all_results.append(result)

        return all_results

    @staticmethod
    def log_results(results, dataset_name, steps, logging=True, print=True, num_fold=None):
        # Print a header
        header = "\n\n"
        header += BUSH_SEP + "\n"
        header += "***************************************************\n"
        if num_fold:
            header += f"***** EVALUATION | FOLD: {num_fold} | {dataset_name.upper()} SET | AFTER {steps} BATCHES *****\n"
        else:
            header += f"***** EVALUATION | {dataset_name.upper()} SET | AFTER {steps} BATCHES *****\n"
        header += "***************************************************\n"
        header += BUSH_SEP + "\n"
        logger.info(header)

        for head_num, head in enumerate(results):  # noqa: B007
            logger.info("\n _________ {} _________".format(head["task_name"]))
            for metric_name, metric_val in head.items():
                # log with ML framework (e.g. Mlflow)
                if logging:  # noqa: SIM102
                    if metric_name not in ["preds", "labels"] and not metric_name.startswith("_"):  # noqa: SIM102
                        if isinstance(metric_val, numbers.Number):
                            MlLogger.log_metrics(
                                metrics={f"{dataset_name}_{metric_name}_{head['task_name']}": metric_val},
                                step=steps,
                            )
                # print via standard python logger
                if print:
                    if metric_name == "report":
                        if isinstance(metric_val, str) and len(metric_val) > 8000:
                            metric_val = metric_val[:7500] + "\n ............................. \n" + metric_val[-500:]
                        logger.info(f"{metric_name}: \n {metric_val}")
                    else:
                        if metric_name not in ["preds", "labels"] and not metric_name.startswith("_"):
                            logger.info(f"{metric_name}: {metric_val}")
