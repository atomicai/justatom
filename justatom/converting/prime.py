from justatom.converting.mask import IConverter
from typing import List, Dict


class TRIOConverter(IConverter):
    """TRIplet loss converter for futher fine-tuning..."""

    def convert(self, x: List[Dict]):
        pass


class M1LMConverter(IConverter):
    def convert(self, x: List[Dict]):
        pass


class M2LMConverter(IConverter):
    #
    def convert(self, x: List[Dict]):
        pass
