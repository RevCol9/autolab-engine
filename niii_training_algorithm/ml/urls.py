from django.urls import re_path, include
from rest_framework.routers import DefaultRouter

from . import views

router = DefaultRouter(trailing_slash=False)
router.register(r"ml", views.EndpointViewSet, basename="ml")
router.register(r"mlalgorithms", views.MLAlgorithmViewSet, basename="mlalgorithms")
router.register(r"mlalgorithmstatuses", views.MLAlgorithmStatusViewSet, basename="mlalgorithmstatuses")
router.register(r"mlrequests", views.MLRequestViewSet, basename="mlrequests")

urlpatterns = [
    re_path(r"^api/v1/", include(router.urls)),
    re_path(r"^api/v1/(?P<endpoint_name>.+)/predict$", views.PredictView.as_view(), name="predict"),
    re_path(r"^api/v1/(?P<endpoint_name>.+)/detect$", views.DetectView.as_view(), name="detect"),
    re_path(r"^api/v1/(?P<endpoint_name>.+)/classify$", views.ClassifyView.as_view(), name="classify"),
    re_path(r"^api/v1/(?P<endpoint_name>.+)/segment$", views.SegmentView.as_view(), name="instanceSegment"),
    re_path(r"^api/v1/(?P<endpoint_name>.+)/train/detection$", views.TrainDetectionView.as_view(), name="train_detection_model"),
    #url(r"^api/v1/(?P<endpoint_name>.+)/stoptrain/detection$", views.TrainDetectionView.as_view(actions={'stop'}), name="stop_training_detection_model"),
    #url(r"^api/v1/(?P<endpoint_name>.+)/train/classification$", views.TrainDetectionView.as_view(), name="train_classification_model"),
    re_path(r"^api/v1/(?P<endpoint_name>.+)/train/segmentation$", views.TrainSegmentationView.as_view(), name="train_instanceSegmentation_model"),
    re_path(r"^api/v1/uploadSamModel", views.LoadSamModel.as_view(), name="uploadSamModel"),
    re_path(r"^api/v1/autolabel", views.SamAutomatedLabel.as_view(), name="SamPredict"),
]
