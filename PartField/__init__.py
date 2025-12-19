from .inference_feature import inference_feature
from .part_clustering import part_clustering

def PartField_segmentation():
    inference_feature()
    part_clustering()

if __name__ == '__main__':
    PartField_segmentation()