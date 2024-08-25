from justatom.converting.mask import IConverter


class TRIOConverter(IConverter):
    """TRIplet loss converter for futher fine-tuning..."""

    def convert(self, x: list[dict]):
        pass


class M1LMConverter(IConverter):
    def convert(self, x: list[dict]):
        pass


class M2LMConverter(IConverter):
    #
    def convert(self, x: list[dict]):
        pass
