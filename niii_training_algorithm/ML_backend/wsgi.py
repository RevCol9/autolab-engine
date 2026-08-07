import os

from django.core.wsgi import get_wsgi_application

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'ML_backend.settings')

application = get_wsgi_application()


# ML registry
import inspect
from algorithm_model.registry import MLRegistry
from algorithm_model.yolo.yolo_detect import YoloDetector
from algorithm_model.yolo.yolo_classify import YoloClassifier
from algorithm_model.yolo.yolo_segment import YoloSegmentation
#from algorithm_model.Unet.Unet_segment import UnetDetector


try:
    registry = MLRegistry() # create ML registry

    yolo = YoloDetector()
    registry.add_algorithm(endpoint_name="yolo_detector",
                            algorithm_object=yolo,
                            algorithm_name="yolo detector",
                            algorithm_status="production",
                            algorithm_version="0.0.1",
                            owner="NIII",
                            algorithm_description="Yolov8",
                            )

    yolo = YoloClassifier()
    registry.add_algorithm(endpoint_name="yolo_classifier",
                            algorithm_object=yolo,
                            algorithm_name="yolo classifier",
                            algorithm_status="production",
                            algorithm_version="0.0.1",
                            owner="NIII",
                            algorithm_description="Yolov8 classifier",
                            )

    yolo = YoloSegmentation()
    registry.add_algorithm(endpoint_name="yolo_segmentation",
                            algorithm_object=yolo,
                            algorithm_name="yolo segmentation",
                            algorithm_status="production",
                            algorithm_version="0.0.1",
                            owner="NIII",
                            algorithm_description="Yolov8 segmentation",
                            )

except Exception as e:
    print("Exception while loading the algorithms to the registry,", str(e))