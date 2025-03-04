import logging
import os
from pathlib import Path

from torch import nn

from farm.modeling.language_model import LanguageModel
from farm.modeling.prediction_head import PredictionHead, TextSimilarityHead
from farm.utils import WANDBLogger as MlLogger

logger = logging.getLogger(__name__)


class BaseBiAdaptiveModel:
    """
    Base Class for implementing AdaptiveModel with frameworks like PyTorch and ONNX.
    """

    subclasses = {}

    def __init_subclass__(cls, **kwargs):
        """This automatically keeps track of all available subclasses.
        Enables generic load() for all specific AdaptiveModel implementation.
        """
        super().__init_subclass__(**kwargs)
        cls.subclasses[cls.__name__] = cls

    def __init__(self, prediction_heads):
        self.prediction_heads = prediction_heads

    @classmethod
    def load(cls, **kwargs):
        """
        Load corresponding AdaptiveModel Class(AdaptiveModel/ONNXAdaptiveModel) based on the
        files in the load_dir.

        :param kwargs: arguments to pass for loading the model.
        :return: instance of a model
        """
        if (Path(kwargs["load_dir"]) / "model.onnx").is_file():
            model = cls.subclasses["ONNXBiAdaptiveModel"].load(**kwargs)
        else:
            model = cls.subclasses["BiAdaptiveModel"].load(**kwargs)
        return model

    def logits_to_preds(self, logits, **kwargs):
        """
        Get predictions from all prediction heads.

        :param logits: logits, can vary in shape and type, depending on task
        :type logits: object
        :param label_maps: Maps from label encoding to label string
        :param label_maps: dict
        :return: A list of all predictions from all prediction heads
        """
        all_preds = []
        # collect preds from all heads
        for head, logits_for_head in zip(self.prediction_heads, logits, strict=False):
            preds = head.logits_to_preds(logits=logits_for_head, **kwargs)
            all_preds.append(preds)
        return all_preds

    def formatted_preds(self, logits, language_model1, language_model2, **kwargs):
        """
        Format predictions to strings for inference output

        :param logits: model logits
        :type logits: torch.tensor
        :param kwargs: placeholder for passing generic parameters
        :type kwargs: object
        :return: predictions in the right format
        """
        n_heads = len(self.prediction_heads)

        if n_heads == 1:
            preds_final = []
            # This try catch is to deal with the fact that sometimes we collect preds before passing it to
            # formatted_preds (see Inferencer._get_predictions_and_aggregate()) and sometimes we don't
            # (see Inferencer._get_predictions())
            try:
                preds = kwargs["preds"]
                temp = [y[0] for y in preds]
                preds_flat = [item for sublist in temp for item in sublist]
                kwargs["preds"] = preds_flat
            except KeyError:
                kwargs["preds"] = None
            head = self.prediction_heads[0]
            logits_for_head = logits[0]
            preds = head.formatted_preds(logits=logits_for_head, **kwargs)
            # TODO This is very messy - we need better definition of what the output should look like
            if type(preds) == list:  # noqa: E721
                preds_final += preds
            elif type(preds) == dict and "predictions" in preds:  # noqa: E721
                preds_final.append(preds)
        return preds_final

    def connect_heads_with_processor(self, tasks, require_labels=True):
        """
        Populates prediction head with information coming from tasks.

        :param tasks: A dictionary where the keys are the names of the tasks and the values are the details of the task (e.g. label_list, metric, tensor name)
        :param require_labels: If True, an error will be thrown when a task is not supplied with labels)
        :return:
        """  # noqa: E501

        for head in self.prediction_heads:
            head.label_tensor_name = tasks[head.task_name]["label_tensor_name"]
            label_list = tasks[head.task_name]["label_list"]
            if not label_list and require_labels:
                raise Exception(f"The task '{head.task_name}' is missing a valid set of labels")
            label_list = tasks[head.task_name]["label_list"]
            head.label_list = label_list
            num_labels = len(label_list)  # noqa: F841
            head.metric = tasks[head.task_name]["metric"]

    @classmethod
    def _get_prediction_head_files(cls, load_dir, strict=True):
        load_dir = Path(load_dir)
        files = os.listdir(load_dir)
        config_files = [load_dir / f for f in files if "config.json" in f and "prediction_head" in f]
        # sort them to get correct order in case of multiple prediction heads
        config_files.sort()
        return config_files


def loss_per_head_sum(loss_per_head, global_step=None, batch=None):
    """
    Input: loss_per_head (list of tensors), global_step (int), batch (dict)
    Output: aggregated loss (tensor)
    """
    return sum(loss_per_head)


class BiAdaptiveModel(nn.Module, BaseBiAdaptiveModel):
    """PyTorch implementation containing all the modelling needed for your NLP task. Combines 2 language
    models for representation of 2 sequences and a prediction head. Allows for gradient flow back to the 2 language model components."""  # noqa: E501

    def __init__(
        self,
        language_model1,
        language_model2,
        prediction_heads,
        embeds_dropout_prob=0.1,
        device="cuda",
        lm1_output_types=["per_sequence"],  # noqa: B006
        lm2_output_types=["per_sequence"],  # noqa: B006
        loss_aggregation_fn=None,
    ):
        """
        :param language_model1: Any model that turns token ids into vector representations
        :type language_model1: LanguageModel
        :param language_model2: Any model that turns token ids into vector representations
        :type language_model2: LanguageModel
        :param prediction_heads: A list of models that take 2 sequence embeddings and return logits for a given task
        :type prediction_heads: list
        :param embeds_dropout_prob: The probability that a value in the embeddings returned by any of the 2
           language model will be zeroed.
        :param embeds_dropout_prob: float
        :param lm1_output_types: How to extract the embeddings from the final layer of the first language model. When set
                                to "per_token", one embedding will be extracted per input token. If set to
                                "per_sequence", a single embedding will be extracted to represent the full
                                input sequence. Can either be a single string, or a list of strings,
                                one for each prediction head.
        :type lm1_output_types: list or str
        :param lm2_output_types: How to extract the embeddings from the final layer of the second language model. When set
                                to "per_token", one embedding will be extracted per input token. If set to
                                "per_sequence", a single embedding will be extracted to represent the full
                                input sequence. Can either be a single string, or a list of strings,
                                one for each prediction head.
        :type lm2_output_types: list or str
        :param device: The device on which this model will operate. Either "cpu" or "cuda".
        :param loss_aggregation_fn: Function to aggregate the loss of multiple prediction heads.
                                    Input: loss_per_head (list of tensors), global_step (int), batch (dict)
                                    Output: aggregated loss (tensor)
                                    Default is a simple sum:
                                    `lambda loss_per_head, global_step=None, batch=None: sum(tensors)`
                                    However, you can pass more complex functions that depend on the
                                    current step (e.g. for round-robin style multitask learning) or the actual
                                    content of the batch (e.g. certain labels)
                                    Note: The loss at this stage is per sample, i.e one tensor of
                                    shape (batchsize) per prediction head.
        :type loss_aggregation_fn: function
        """

        super(BiAdaptiveModel, self).__init__()  # noqa: UP008
        self.device = device
        self.language_model1 = language_model1.to(device)
        self.lm1_output_dims = language_model1.get_output_dims()
        self.language_model2 = language_model2.to(device)
        self.lm2_output_dims = language_model2.get_output_dims()
        self.dropout1 = nn.Dropout(embeds_dropout_prob)
        self.dropout2 = nn.Dropout(embeds_dropout_prob)
        self.prediction_heads = nn.ModuleList([ph.to(device) for ph in prediction_heads])
        self.lm1_output_types = [lm1_output_types] if isinstance(lm1_output_types, str) else lm1_output_types
        self.lm2_output_types = [lm2_output_types] if isinstance(lm2_output_types, str) else lm2_output_types
        self.log_params()
        # default loss aggregation function is a simple sum (without using any of the optional params)
        if not loss_aggregation_fn:
            loss_aggregation_fn = loss_per_head_sum
        self.loss_aggregation_fn = loss_aggregation_fn

    def save(self, save_dir, lm1_name="query", lm2_name="passage"):
        """
        Saves the 2 language model weights and respective config_files in directories lm1 and lm2 within save_dir.

        :param save_dir: path to save to
        :type save_dir: Path
        """
        os.makedirs(save_dir, exist_ok=True)
        if not os.path.exists(Path.joinpath(save_dir, Path(lm1_name))):
            os.makedirs(Path.joinpath(save_dir, Path(lm1_name)))
        if not os.path.exists(Path.joinpath(save_dir, Path(lm2_name))):
            os.makedirs(Path.joinpath(save_dir, Path(lm2_name)))
        self.language_model1.save(Path.joinpath(save_dir, Path(lm1_name)))
        self.language_model2.save(Path.joinpath(save_dir, Path(lm2_name)))
        for i, ph in enumerate(self.prediction_heads):
            logger.info("prediction_head saving")
            ph.save(save_dir, i)

    @classmethod
    def load(cls, load_dir, device, strict=False, lm1_name="query", lm2_name="passage", processor=None):
        """
        Loads a BiAdaptiveModel from a directory. The directory must contain:

        * directory "lm1_name" with following files:
            -> language_model.bin
            -> language_model_config.json
        * directory "lm2_name" with following files:
            -> language_model.bin
            -> language_model_config.json
        * prediction_head_X.bin  multiple PH possible
        * prediction_head_X_config.json
        * processor_config.json config for transforming input
        * vocab.txt vocab file for language model, turning text to Wordpiece Token
        * special_tokens_map.json

        :param load_dir: location where adaptive model is stored
        :type load_dir: Path
        :param device: to which device we want to sent the model, either cpu or cuda
        :type device: torch.device
        :param lm1_name: the name to assign to the first loaded language model(for encoding queries)
        :type lm1_name: str
        :param lm2_name: the name to assign to the second loaded language model(for encoding context/passages)
        :type lm2_name: str
        :param strict: whether to strictly enforce that the keys loaded from saved model match the ones in
                       the PredictionHead (see torch.nn.module.load_state_dict()).
                       Set to `False` for backwards compatibility with PHs saved with older version of FARM.
        :type strict: bool
        :param processor: populates prediction head with information coming from tasks
        :type processor: Processor
        """
        # Language Model
        if lm1_name:  # noqa: SIM108
            language_model1 = LanguageModel.load(os.path.join(load_dir, lm1_name))
        else:
            language_model1 = LanguageModel.load(load_dir)
        if lm2_name:  # noqa: SIM108
            language_model2 = LanguageModel.load(os.path.join(load_dir, lm2_name))
        else:
            language_model2 = LanguageModel.load(load_dir)

        # Prediction heads
        ph_config_files = cls._get_prediction_head_files(load_dir)
        prediction_heads = []
        ph_output_type = []
        for config_file in ph_config_files:
            head = PredictionHead.load(config_file, strict=False, load_weights=False)
            prediction_heads.append(head)
            ph_output_type.append(head.ph_output_type)

        model = cls(language_model1, language_model2, prediction_heads, 0.1, device)
        if processor:
            model.connect_heads_with_processor(processor.tasks)

        return model

    def logits_to_loss_per_head(self, logits, **kwargs):
        """
        Collect losses from each prediction head.

        :param logits: logits, can vary in shape and type, depending on task.
        :type logits: object
        :return: The per sample per prediciton head loss whose first two dimensions have length n_pred_heads, batch_size
        """
        all_losses = []
        for head, logits_for_one_head in zip(self.prediction_heads, logits, strict=False):
            # check if PredictionHead connected to Processor
            assert hasattr(head, "label_tensor_name"), (
                f"Label_tensor_names are missing inside the {head.task_name} Prediction Head. Did you connect the model"
                " with the processor through either 'model.connect_heads_with_processor(processor.tasks)'"
                " or by passing the processor to the Adaptive Model?"
            )
            all_losses.append(head.logits_to_loss(logits=logits_for_one_head, **kwargs))
        return all_losses

    def logits_to_loss(self, logits, global_step=None, **kwargs):
        """
        Get losses from all prediction heads & reduce to single loss *per sample*.

        :param logits: logits, can vary in shape and type, depending on task
        :type logits: object
        :param global_step: number of current training step
        :type global_step: int
        :param kwargs: placeholder for passing generic parameters.
                       Note: Contains the batch (as dict of tensors), when called from Trainer.train().
        :type kwargs: object
        :return loss: torch.tensor that is the per sample loss (len: batch_size)
        """
        all_losses = self.logits_to_loss_per_head(logits, **kwargs)
        # This aggregates the loss per sample across multiple prediction heads
        # Default is sum(), but you can configure any fn that takes [Tensor, Tensor ...] and returns [Tensor]
        loss = self.loss_aggregation_fn(all_losses, global_step=global_step, batch=kwargs)
        return loss

    def prepare_labels(self, **kwargs):
        """
        Label conversion to original label space, per prediction head.

        :param label_maps: dictionary for mapping ids to label strings
        :type label_maps: dict[int:str]
        :return: labels in the right format
        """
        all_labels = []
        # for head, label_map_one_head in zip(self.prediction_heads):
        #     labels = head.prepare_labels(label_map=label_map_one_head, **kwargs)
        #     all_labels.append(labels)
        for head in self.prediction_heads:
            labels = head.prepare_labels(**kwargs)
            all_labels.append(labels)
        return all_labels

    def forward(self, **kwargs):
        """
        Push data through the whole model and returns logits. The data will propagate through
        the first language model and second language model based on the tensor names and both the
        encodings through each of the attached prediction heads.

        :param kwargs: Holds all arguments that need to be passed to both the language models and prediction head(s).
        :return: all logits as torch.tensor or multiple tensors.
        """

        # Run forward pass of both language models
        pooled_output = self.forward_lm(**kwargs)

        # Run forward pass of (multiple) prediction heads using the output from above
        all_logits = []
        if len(self.prediction_heads) > 0:
            for head, lm1_out, lm2_out in zip(self.prediction_heads, self.lm1_output_types, self.lm2_output_types, strict=False):
                # Choose relevant vectors from LM as output and perform dropout
                if pooled_output[0] is not None:
                    if lm1_out == "per_sequence" or lm1_out == "per_sequence_continuous":
                        output1 = self.dropout1(pooled_output[0])
                    else:
                        raise ValueError(f"Unknown extraction strategy from BiAdaptive language_model1: {lm1_out}")
                else:
                    output1 = None

                if pooled_output[1] is not None:
                    if lm2_out == "per_sequence" or lm2_out == "per_sequence_continuous":
                        output2 = self.dropout2(pooled_output[1])
                    else:
                        raise ValueError(f"Unknown extraction strategy from BiAdaptive language_model2: {lm2_out}")
                else:
                    output2 = None

                embedding1, embedding2 = head(output1, output2)
                all_logits.append(tuple([embedding1, embedding2]))
        else:
            # just return LM output (e.g. useful for extracting embeddings at inference time)
            all_logits.append(pooled_output)

        return all_logits

    def forward_lm(self, **kwargs):
        """
        Forward pass for the BiAdaptive model.

        :param kwargs:
        :return: 2 tensors of pooled_output from the 2 language models
        """
        pooled_output = [None, None]
        if "query_input_ids" in kwargs.keys():  # noqa: SIM118
            # if self.language_model1.get_language_model_class(self.language_model1.name) == 'DistilBert':
            #     pooled_output1, hidden_states1 = self.language_model1(input_ids=kwargs['query_input_ids'], padding_mask=kwargs['query_attention_mask'])  # noqa: E501
            pooled_output1, hidden_states1 = self.language_model1(**kwargs)
            pooled_output[0] = pooled_output1
        if "passage_input_ids" in kwargs.keys():  # noqa: SIM118
            pooled_output2, hidden_states2 = self.language_model2(**kwargs)
            pooled_output[1] = pooled_output2

        return tuple(pooled_output)

    def log_params(self):
        """
        Logs paramteres to generic logger MlLogger
        """
        params = {
            "lm1_type": self.language_model1.__class__.__name__,
            "lm1_name": self.language_model1.name,
            "lm1_output_types": ",".join(self.lm1_output_types),
            "lm2_type": self.language_model2.__class__.__name__,
            "lm2_name": self.language_model2.name,
            "lm2_output_types": ",".join(self.lm2_output_types),
            "prediction_heads": ",".join([head.__class__.__name__ for head in self.prediction_heads]),
        }
        try:
            MlLogger.log_params(params)
        except Exception as e:
            logger.warning(f"ML logging didn't work: {e}")

    def verify_vocab_size(self, vocab_size1, vocab_size2):
        """Verifies that the model fits to the tokenizer vocabulary.
        They could diverge in case of custom vocabulary added via tokenizer.add_tokens()"""

        model1_vocab_len = self.language_model1.model.resize_token_embeddings(new_num_tokens=None).num_embeddings

        msg = (
            f"Vocab size of tokenizer {vocab_size1} doesn't match with model {model1_vocab_len}. "
            "If you added a custom vocabulary to the tokenizer, "
            "make sure to supply 'n_added_tokens' to LanguageModel.load() and BertStyleLM.load()"
        )
        assert vocab_size1 == model1_vocab_len, msg

        model2_vocab_len = self.language_model2.model.resize_token_embeddings(new_num_tokens=None).num_embeddings

        msg = (
            f"Vocab size of tokenizer {vocab_size1} doesn't match with model {model2_vocab_len}. "
            "If you added a custom vocabulary to the tokenizer, "
            "make sure to supply 'n_added_tokens' to LanguageModel.load() and BertStyleLM.load()"
        )
        assert vocab_size2 == model2_vocab_len, msg

    def get_language(self):
        return self.language_model1.language, self.language_model2.language

    def convert_to_transformers(self):
        from transformers import AutoModel, DPRContextEncoder, DPRQuestionEncoder

        if len(self.prediction_heads) != 1:
            raise ValueError(
                f"Currently conversion only works for models with a SINGLE prediction head. "
                f"Your model has {len(self.prediction_heads)}"
            )

        if self.prediction_heads[0].model_type == "text_similarity":
            # init model
            if "dpr" in self.language_model1.model.config.model_type:
                transformers_model1 = DPRQuestionEncoder(config=self.language_model1.model.config)
            else:
                transformers_model1 = AutoModel.from_config(config=self.language_model1.model.config)
            if "dpr" in self.language_model2.model.config.model_type:
                transformers_model2 = DPRContextEncoder(config=self.language_model2.model.config)
            else:
                transformers_model2 = AutoModel.from_config(config=self.language_model2.model.config)

            # transfer weights for language model + prediction head
            setattr(transformers_model1, transformers_model1.base_model_prefix, self.language_model1.model)
            setattr(transformers_model2, transformers_model2.base_model_prefix, self.language_model2.model)
            logger.warning("No prediction head weights are required for DPR")

        else:
            raise NotImplementedError(
                f"FARM -> Transformers conversion is not supported yet for"
                f" prediction heads of type {self.prediction_heads[0].model_type}"
            )
        pass

        return transformers_model1, transformers_model2

    @classmethod
    def convert_from_transformers(
        cls, model_name_or_path1, model_name_or_path2, device, task_type, processor=None, similarity_function="dot_product"
    ):
        """
        Load a (downstream) model from huggingface's transformers format. Use cases:
         - continue training in FARM (e.g. take a squad QA model and fine-tune on your own data)
         - compare models without switching frameworks
         - use model directly for inference

        :param model_name_or_path1: local path of a saved model or name of a public one for Question Encoder
                                              Exemplary public names:
                                              - facebook/dpr-question_encoder-single-nq-base
                                              - deepset/bert-large-uncased-whole-word-masking-squad2
        :param model_name_or_path2: local path of a saved model or name of a public one for Context/Passage Encoder
                                      Exemplary public names:
                                      - facebook/dpr-ctx_encoder-single-nq-base
                                      - deepset/bert-large-uncased-whole-word-masking-squad2
        :param device: "cpu" or "cuda"
        :param task_type: 'text_similarity'
                          More tasks coming soon ...
        :param processor: populates prediction head with information coming from tasks
        :type processor: Processor
        :return: AdaptiveModel
        """
        lm1 = LanguageModel.load(pretrained_model_name_or_path=model_name_or_path1, language_model_class="DPRQuestionEncoder")
        lm2 = LanguageModel.load(pretrained_model_name_or_path=model_name_or_path2, language_model_class="DPRContextEncoder")
        prediction_head = TextSimilarityHead(similarity_function=similarity_function)
        # TODO Infer type of head automatically from config
        if task_type == "text_similarity":
            bi_adaptive_model = cls(
                language_model1=lm1,
                language_model2=lm2,
                prediction_heads=[prediction_head],
                embeds_dropout_prob=0.1,
                lm1_output_types=["per_sequence"],
                lm2_output_types=["per_sequence"],
                device=device,
            )
        else:
            raise NotImplementedError(
                f"Huggingface's transformer models of type {task_type} are not supported yet for BiAdaptive Models"
            )

        if processor:
            bi_adaptive_model.connect_heads_with_processor(processor.tasks)

        return bi_adaptive_model
