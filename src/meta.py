from pyrocko.guts import TBase, ValidationError, String

MAG = dict(k=3, M=6, G=9, T=12, P=15)


class Path(String):
    ...


class DataSize(String):
    dummy_for = int

    class __T(TBase):

        def regularize_extra(self, val):
            if isinstance(val, int):
                return val

            if isinstance(val, str):
                size = val.strip()
                try:
                    v = float(val.strip(''.join(MAG.keys())))
                except ValueError:
                    raise ValidationError('cannot interpret %s', val)
                if not v:
                    raise ValidationError('cannot interpret %s', val)

                for suffix, m in MAG.items():
                    if size.endswith(suffix):
                        return int(v*10**m)

                return int(v)

            raise ValidationError('cannot interpret data size %s' % size)

        def to_save(self, val):
            if isinstance(val, int):
                return val
            return str(val)
