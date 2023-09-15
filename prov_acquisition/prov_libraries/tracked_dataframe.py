from typing import Iterable


def add_methods_to_child_class(child, base, exceptions, tracker, tracker_id):
    """
    Adds methods from the base class to the child class, tracking their provenance.
    Ignores methods listed in exceptions.

    :param child: The child class to which methods will be added.
    :param base: The base class from which methods will be taken.
    :param exceptions: List of methods to be ignored.
    :param tracker: Provenance Tracker.
    :param tracker_id: Tracker ID.

    :return: None
    """

    for field_name, field in base.__dict__.items():
        if callable(field) and field_name not in exceptions and not isinstance(field, Iterable):
            setattr(child, field_name, tracker._wrapper_track_provenance(field, tracker_id))


class TrackedDataFrameMeta(type):
    """
    Defines the metaclass for TrackedDataFrame.

    """

    def __new__(cls, name, bases, dct, tracker, tracker_id: str):
        """
        Every method (except exceptions) will be encapsulated.
        The wrapper function will handle tracking the provenance.

        :param name: Name of the class.
        :param bases: Base classes.
        :param dct: Class attributes.
        :param tracker: Provenance Tracker.
        :param tracker_id: Tracker ID.

        :return: The new child class.
        """

        child = super().__new__(cls, name, bases, dct)

        setattr(child, 'tracker_id', tracker_id)

        exceptions = ['__init__', '_constructor_sliced', '_get_item_cache', '_clear_item_cache', '_ixs',
                      '_box_col_values', 'iterrows', '__repr__', '_info_repr', 'to_string', '__len__', 'itertuples',
                      'to_dict', '__getitem__', '_maybe_cache_changed', '_append', '_set_item', '_sanitize_column',
                      '_ensure_valid_index', '_set_item_mgr', '_iset_item_mgr', '_cmp_method', '_dispatch_frame_op',
                      '_construct_result', '_setitem_frame', 'isna', 'to_numpy', 'values', 'corr', 'isnull', 'nunique',
                      'select_dtypes', 'items', 'assign']

        for base in bases:
            add_methods_to_child_class(child, base, exceptions, tracker, tracker_id)

        return child


class TrackedDataFrameGroupByMeta(type):
    #TODO
    def __new__(cls, name, bases, dct, tracker, tracker_id: str):
        child = super().__new__(cls, name, bases, dct)

        setattr(child, 'tracker_id', tracker_id)

        exceptions = ['__getattribute__', '__dir__', '_get_data_to_aggregate', '__get_prov_from_aggregate', 'aggregate']

        for base in bases:
            add_methods_to_child_class(child, base, exceptions, tracker, tracker_id)

        return child