from rethinkdb import r

from justatom.storing.mask import IEVENTDocStore


class IReDocStore(IEVENTDocStore):
    def __init__(self, host, port, **props):
        super().__init__()
        self.client = r.connect(host=host, port=port)

    async def add_event(self, e):  #
        pass

    async def add_user(self, username, creds, uuid):
        pass

    async def del_user(self, uuid):
        pass

    async def add_document(self, doc):
        pass

    async def del_document(self, uuid):
        pass


__all__ = ["IReDocStore"]
