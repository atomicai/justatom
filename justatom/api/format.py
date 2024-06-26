from loguru import logger
from typing import Union, List, Optional, Dict
import io
import os
import fire
import polars as pl
import copy
from justatom.tooling import stl
from justatom.etc.format import maybe_number
from justatom.etc.lazy_imports import LazyImport
from pathlib import Path

with LazyImport("Run 'pip install docx2txt'") as docx_import:
    import docx2txt

"""
This API point labels incoming .docx document correctly as long as it follows the structure:
(*) ?<title>?<type>:<query>:<context>:<answer>
--- (1) <title> | The book title. Could be ommited, hence wrapped around in `?`
--- (2) <type> | One of the two possible variants. (i) - 'extractive' or (ii) 'extractive:dialogue'.
--- (3) <query> | The query that could be answered using <context> as provided micro-knowledge db.
--- (4) <context> | The context to help answer the <query>.
--- (5) <answer> | The answer to the <query> given <context>.
"""


class IState:

    def __init__(self, name: str = None):
        self.name = name

    def __hash__(self):
        return hash(self.name) if self.name is not None else str(self)

    def __str__(self):
        return self.__class__.__name__

    def observe(self, data: str):
        data = data.strip()
        if data.startswith("Человек он благонадежный и обеспеченный") or data.startswith("10") or data.startswith("11"):
            print()
        _observable = data.strip().lower()
        if _observable.startswith("title:"):
            return self.process(data, prefix_to_remove="title:", state_to_move=TITLEState)
        if _observable.startswith(
            (
                "extractive:dialogue",
                "еxtractive: dialogue",
                "extractive: dialogue",
                "extractive",
                "еxtractive",
                "extractive:",
                "abstractive",
                "abstractive",
            )
        ):
            return self.process(data, prefix_to_remove="", state_to_move=TYPEState)
        elif _observable.startswith("query:"):
            return self.process(data, prefix_to_remove="query:", state_to_move=QUERYState)
        elif _observable.startswith("context:"):
            return self.process(data, prefix_to_remove="context:", state_to_move=CONTEXTState)
        elif _observable.startswith("answer:"):
            return self.process(data, prefix_to_remove="answer:", state_to_move=ANSWERState)
        elif _observable.startswith("аnswer:"):
            return self.process(data, prefix_to_remove="аnswer:", state_to_move=ANSWERState)
        elif maybe_number(_observable):  # `maybe_real`
            return self.process(data, prefix_to_remove="", state_to_move=ENUMState)

        # Right now we know exactly that query class is one of `BUFFERState`
        # To determine the exact

        if self.__class__.__name__ in ("QUERYState", "QUERYBufferState"):
            return self.process(data, prefix_to_remove="", state_to_move=QUERYBufferState)
        elif self.__class__.__name__ in ("CONTEXTState", "CONTEXTBufferState"):
            return self.process(data, prefix_to_remove="", state_to_move=CONTEXTBufferState)
        elif self.__class__.__name__ in ("ANSWERState", "ANSWERBufferState"):
            return self.process(data, prefix_to_remove="", state_to_move=ANSWERBufferState)
        raise Exception(f"Couldn't react to the following [{data}] input given current state {self.__class__.__name__}")

    @property
    def belongs_to(self):
        return self.name
    
    @property
    def derives_as(self):
        return None

    def next(self, data=None):
        # Can we yield each state here sequentially?
        # TODO:
        # How to props the prev data sequentially?
        formatted_data, obs = self.observe(data)
        if obs in self.transitions:
            return formatted_data, self.transitions[obs]()
        raise ValueError(f"Got unexpected input [{obs}] in state <{self.name}>")

    def process(self, x, prefix_to_remove, state_to_move):
        result = x[len(prefix_to_remove) :].strip()
        logger.info(f"PROCESSING - NEW STATE [{state_to_move.__name__}] - on LINE {result}")
        return result, state_to_move.__name__


class STARTState(IState):

    def __init__(self):
        super().__init__(name=self.__class__.__name__)
        self.transitions = dict(TITLEState=TITLEState)


class TITLEState(IState):

    # TITLEState is the starting state

    def __init__(self):
        super().__init__(name=self.__class__.__name__)
        self.transitions = dict(ENUMState=ENUMState)


class ENUMState(IState):

    # ENUMState is the state you enter upon reading numbered order of the snippet
    # e.g. "22." => The twenty second example follows
    # e.g. "extractive:dialogue"

    def __init__(self):
        super().__init__(name=self.__class__.__name__)
        self.transitions = dict(TYPEState=TYPEState)


class TYPEState(IState):

    # TYPEState is the state you enter upon reading format line: <extractive> or <extractive:dialogue>
    # From this state the <query> should come next

    def __init__(self):
        super().__init__(name=self.__class__.__name__)
        self.transitions = dict(QUERYState=QUERYState)


class QUERYState(IState):

    # QUERYState is the state you enter upon reading query line that follows pattern: <query: What are the rules of the hunger games?>
    # From this state you can move either
    # (1) directly to the context <context: In punishment for the uprising...>
    # (2) move to the continuation of the query if it's quite long => `QUERYBufferState`

    def __init__(self):
        super().__init__(name=self.__class__.__name__)
        self.transitions = dict(CONTEXTState=CONTEXTState, QUERYBufferState=QUERYBufferState)


class QUERYBufferState(IState):

    # QUERYBufferState is the optional state which means you may or may not enter it. Triggering case is upon reading the query line
    #  <query:What are the\n // QUERYState
    #  rules of the hunger games> // QUERYBufferState
    
    group = "QUERYState"

    def __init__(self):
        super().__init__(name=self.__class__.__name__)
        self.transitions = dict(CONTEXTState=CONTEXTState, QUERYBufferState=QUERYBufferState)

    @property
    def belongs_to(self):
        return "BUFFER"
    
    @property
    def derives_as(self):
        return QUERYState


class CONTEXTState(IState):

    # CONTEXTState is the state you enter upon reading context line that follows pattern: <context: The rules of the hunger games ...>
    # From this state you can move either
    # (1) directly to the answer <answer: Twenty four tributes, male and female from each district>
    # (2) move to the continuation of the context if it's quite long => `BUFFERState`

    def __init__(self):
        super().__init__(name=self.__class__.__name__)
        self.transitions = dict(ANSWERState=ANSWERState, CONTEXTBufferState=CONTEXTBufferState)


class CONTEXTBufferState(IState):

    # CONTEXTBufferState is the optional state meaning you may or may not enter it. Triggering case is upon reading context line
    # <context:The rules of the\n // CONTEXTState
    # hunger games are simple. In punishment ... > // CONTEXTBufferState
    
    group = "CONTEXTState"

    def __init__(self):
        super().__init__(name=self.__class__.__name__)
        self.transitions = dict(ANSWERState=ANSWERState, CONTEXTBufferState=CONTEXTBufferState)

    @property
    def belongs_to(self):
        return "BUFFER"
    
    @property
    def derives_as(self):
        return CONTEXTState


class ANSWERState(IState):
    # ANSWERState is the state you enter upon reading answer line that follows pattern: <answer: Twenty-four tributes between\n
    # the ages of twelve and eighteen...>
    # From this state you can move either
    # (1) directly to the enum <22.>
    # (2) move to the continuation of the answer if it's quite long => `BUFFERState`

    def __init__(self):
        super().__init__(name=self.__class__.__name__)
        self.transitions = dict(ENUMState=ENUMState, ANSWERBufferState=ANSWERBufferState)


class ANSWERBufferState(IState):

    # ANSWERBufferState is the optional state meaning you may or may not enter it. Triggering case is upon reading answer line
    # <answer: In punishment for the uprising, each of the twelve districts must provide one girl and one boy,\n // ANSWERState
    # called tributes, to participate. The 24 tributes will be imprisoned in a vast outdoor arena\n // ANSWERBUfferState
    # that could hold anything from a burning desert to a frozen wasteland. // ANSWERBufferState
    group = "ANSWERState"

    def __init__(self):
        super().__init__(name=self.__class__.__name__)
        self.transitions = dict(ENUMState=ENUMState, ANSWERBufferState=ANSWERBufferState)

    @property
    def belongs_to(self) -> str:
        return "BUFFER"

    @property
    def derives_as(self) -> str:
        return ANSWERState


def make_one_full_state_chunk():
    pass


def make_one_full_state_chunk():
    pass

def check_and_flush(cur_state: IState, arr: List[str], sample: Dict, schema_mapping: Dict):
    """
    This is triggered upon next state followed by any of `state_i`: `state_i`.belongs_to == "BUFFER"
    """
    if cur_state.derives_as is None:
        return sample
    assert len(arr) > 0, f"The buffer is empty even though you've entered [BUFFER] as [{cur_state}]"
    
    sch_key = schema_mapping[cur_state.group]
    value = "\n".join(arr)
    sample[sch_key] = value
    # flush `arr` to begin next sample clear
    arr.clear()
    return sample
    

def main(one_iterator):
    cur_state: IState = STARTState()
    samples = []
    state_to_sample = dict(
        QUERYState="query", CONTEXTState="context", ANSWERState="answer", TYPEState="format", TITLEState="title"
    )
    cur_state_sample = dict()
    cur_state_buffer = []
    while one_iterator.has_next():
        chunk = one_iterator.next()
        chunk = chunk.strip()
        if chunk == "" or chunk == "\n":
            continue
        formatted_data, next_state = cur_state.next(data=chunk)
        if next_state.belongs_to == "BUFFER":
            if len(cur_state_buffer) <= 0:
                cur_state_buffer.append(cur_state_sample[state_to_sample[str(cur_state)]])
            cur_state_buffer.append(formatted_data)

        if next_state.belongs_to == "QUERYState":
            # previous state was `TYPEState`
            key, data = state_to_sample[str(next_state)], formatted_data
            cur_state_sample[key] = data
            #
            cur_state_sample = check_and_flush(cur_state, cur_state_buffer, cur_state_sample, state_to_sample)
        elif next_state.belongs_to == "CONTEXTState":
            key, data = state_to_sample[str(next_state)], formatted_data
            cur_state_sample[key] = data
            cur_state_sample = check_and_flush(cur_state, arr=cur_state_buffer, sample=cur_state_sample, schema_mapping=state_to_sample)
        elif next_state.belongs_to == "ANSWERState":
            key, data = state_to_sample[str(next_state)], formatted_data
            cur_state_sample[key] = data
            cur_state_sample = check_and_flush(cur_state, arr=cur_state_buffer, sample=cur_state_sample, schema_mapping=state_to_sample)
        elif next_state.belongs_to == "TITLEState":
            key, data = state_to_sample[str(next_state)], formatted_data
            cur_state_sample.clear()
            cur_state_sample[key] = data
        elif next_state.belongs_to == "ENUMState":
            # ... Flush the pipeline to append new sample ...
            cur_state_sample = check_and_flush(cur_state, arr=cur_state_buffer, sample=cur_state_sample, schema_mapping=state_to_sample)
            if cur_state_sample.keys() == set(state_to_sample.values()):
                # This `if` is needed to handle cases when new title has just came in
                # but no samples had been yet processes. E.g.
                # Title:Hunger Games
                # 1
                # `cur_state_sample` had only `title` so we skip
                samples.append(copy.deepcopy(cur_state_sample))
            else:
                logger.info(f"Incomplete sample yet, present fields are {','.join(cur_state_sample.keys())}")
            logger.info(f"LOCDOC - STEP {len(samples)}")
        elif next_state.belongs_to == "TYPEState":
            key, data = state_to_sample[str(next_state)], formatted_data
            cur_state_sample[key] = data
        cur_state = next_state

    if cur_state_sample.keys() == set(state_to_sample.values()):
        cur_state_sample = check_and_flush(cur_state, arr=cur_state_buffer, sample=cur_state_sample, schema_mapping=state_to_sample)
        samples.append(copy.deepcopy(cur_state_sample))

    return samples


def io_wrapper_txt(fp, chunk_size: int = 10_000_000, sep: str = "\n"):
    fp = Path(fp)
    if fp.suffix != ".txt":
        message = f"Calling `io_wrapper_txt` on a file ending with [{fp.suffix}] suffix."
        logger.error(message)
        raise ValueError(message)
    with open(fp) as fin:
        for chunk in stl.chunkify(fin, chunksize=chunk_size, sep=sep):
            yield chunk


def io_wrapper_docx(fp, chunk_size: int = 10_000_000, sep: str = "\n"):
    fp = Path(fp)
    if fp.suffix != ".docx":
        message = f"Calling `io_wrapper_docx` on a file ending with [{fp.suffix}] suffix."
        logger.error(message)
        raise ValueError(message)
    try:
        data = docx2txt.process(fp)
    except:
        message = f"Error reading [{fp}] docx file."
        logger.error(message)
        raise ValueError(message)
    else:
        for chunk in stl.chunkify(io.StringIO(data), chunksize=chunk_size, sep=sep):
            yield chunk


def ignite_io_loaders(fpaths: List[Union[str, Path]]) -> stl.NIterator:
    iterators = []
    for fpath in fpaths:
        fp = Path(fpath)
        if fp.suffix in (".txt"):
            iterator = io_wrapper_txt(fp)
        elif fp.suffix in (".docx"):
            iterator = io_wrapper_docx(fp)
        else:
            continue
        iterators.append(iterator)
    return stl.NIterator(stl.merge_iterators(*iterators))


def parse(filepath_or_dir, extensions: Optional[List[str]] = None, out_path: Optional[str] = None):
    AVAILABLE_EXTENSIONS = [".docx", ".txt"]
    fpath_or_dir = Path(filepath_or_dir)
    if fpath_or_dir.is_dir():
        extensions = set([extensions] if isinstance(extensions, str) else extensions)
        okay_paths = [p for p in fpath_or_dir.iterdir() if p.suffix in extensions]
        # Note that `okay_paths` could be <= 0
        if len(okay_paths) <= 0:
            message = f"Provided `filepath_or_dir` {str(filepath_or_dir)} is a directory but no files found matching \
                        one of [{','.join(AVAILABLE_EXTENSIONS)}]"
    else:
        if not fpath_or_dir.exists():
            message = f"Provided `filepath_or_dir` {str(fpath_or_dir)} doesn't exist"
            logger.error(message)
            raise ValueError(message)
        elif fpath_or_dir.suffix in AVAILABLE_EXTENSIONS:
            okay_paths = [fpath_or_dir]
        else:
            message = f"Provided `filepath_or_dir` {str(fpath_or_dir)} exists but suffix [{fpath_or_dir.suffix}] is not yet \
                        supported. Provide one of [{','.join(AVAILABLE_EXTENSIONS)}]"
            logger.error(message)
            raise ValueError(message)

    make_one_iterator = ignite_io_loaders(okay_paths)
    samples = main(make_one_iterator)
    pl_wrapper = pl.from_dicts(samples)
    logger.info(f"There are {pl_wrapper.shape[0]} samples")

    out_path = (
        Path(os.getcwd()) / ".data" / "outputs" / f"{fpath_or_dir.stem}.csv" if out_path is None else Path(out_path)
    )

    if out_path.suffix != ".csv":
        _wrong_suffix = out_path.suffix
        out_path = out_path.parent / f"{fpath_or_dir.stem}.csv"
        logger.warning(
            f"\
                You provided resulting filepath to have [{_wrong_suffix}] suffix which is not supported.\
                New filepath = [{out_path}]\
            "
        )
    out_path.parent.mkdir(exist_ok=True, parents=True)
    pl_wrapper.write_csv(f"{str(out_path)}")


if __name__ == "__main__":
    fire.Fire(parse)
