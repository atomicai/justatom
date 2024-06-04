from justatom.storing.mask import IDBDocStore
from rethinkdb import r


class IReDocStore(IDBDocStore):

    def __init__(self, host, port, **props):
        super().__init__()
        self.client = r.connect(host=host, port=port)

    def add_event(self, e):
        pass

    def add_user(self, username, creds, uuid):
        pass

    def del_user(self, uuid):
        pass

    def add_document(self, doc):
        pass

    def del_document(self, uuid):
        pass


__all__ = ["IReDocStore"]
