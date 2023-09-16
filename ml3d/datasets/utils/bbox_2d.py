from ...vis import BoundingBox2D

class BBox2D(BoundingBox2D):
    """Class that defines a 2D bounding box (wrapper for BoundingBox2D)"""

    def __init__(self,
                 bbox,
                 label_class,
                 confidence,
                 **kwargs):
        super().__init__(bbox,
                         label_class,
                         confidence,
                         **kwargs)