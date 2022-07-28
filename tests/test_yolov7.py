import unittest

from tests.test_utils import TestConstants

DEVICE = "cpu"

class TestYolov7(unittest.TestCase):
    def test_load_model(self):
        from yolov7 import YOLOv7

        # init model
        model_path = TestConstants.YOLOV7_MODEL_PATH
        yolov7 = YOLOv7(model_path, DEVICE, load_on_init=False)
        yolov7.load_model()

        # check if loaded
        self.assertNotEqual(yolov7.model, None)

    def test_load_model_on_init(self):
        from yolov7 import YOLOv7

        # init model
        model_path = TestConstants.YOLOV7_MODEL_PATH
        yolov7 = YOLOv7(model_path, DEVICE, load_on_init=True)

        # check if loaded
        self.assertNotEqual(yolov7.model, None)

    def test_predict(self):
        from PIL import Image
        from yolov7 import YOLOv7

        # init YOLOV7 model
        model_path = TestConstants.YOLOV7_MODEL_PATH
        yolov7 = YOLOv7(model_path, DEVICE, load_on_init=True)

        # prepare image
        image_path = TestConstants.ZIDANE_IMAGE_PATH
        image = Image.open(image_path)

        # perform inference
        results = yolov7.predict(image, size=640, augment=False)

        # compare
        self.assertEqual(results.n, 1)
        self.assertEqual(len(results.names), 80)
        self.assertEqual(len(results.pred[0]), 4)
        
        # prepare image
        image = image_path

        # perform inference
        results = yolov7.predict(image, size=640, augment=True)

        # compare
        self.assertEqual(results.n, 1)
        self.assertEqual(len(results.names), 80)
        self.assertEqual(len(results.pred[0]), 5)        
        # init YOLOv7-W6 model
        model_path = TestConstants.YOLOv7_W6_MODEL_PATH
        yolov7 = YOLOv7(model_path, DEVICE, load_on_init=True)

        # prepare image
        image_path = TestConstants.BUS_IMAGE_PATH
        image = Image.open(image_path)
        # perform inference
        results = yolov7.predict(image, size=1280, augment=False)

        # compare
        self.assertEqual(results.n, 1)
        self.assertEqual(len(results.names), 80)
        self.assertEqual(len(results.pred[0]), 6)
        
        # prepare image
        image = image_path

        # perform inference
        results = yolov7.predict(image, size=1280, augment=False)

        # compare
        self.assertEqual(results.n, 1)
        self.assertEqual(len(results.names), 80)
        self.assertEqual(len(results.pred[0]), 6)

        # init YOLOV7  model
        model_path = TestConstants.YOLOV7_MODEL_PATH
        yolov7 = YOLOv7(model_path, DEVICE, load_on_init=True)

        # prepare images
        image_path1 = TestConstants.ZIDANE_IMAGE_PATH
        image_path2 = TestConstants.BUS_IMAGE_PATH
        image1 = Image.open(image_path1)
        image2 = Image.open(image_path2)

        # perform inference with multiple images and test augmentation
        results = yolov7.predict([image1, image2], size=1280, augment=True)

        # compare
        self.assertEqual(results.n, 2)
        self.assertEqual(len(results.names), 80)
        self.assertEqual(len(results.pred[0]), 4)
        self.assertEqual(len(results.pred[1]), 5)

        # prepare image
        image_path1 = TestConstants.ZIDANE_IMAGE_PATH
        image_path2 = TestConstants.BUS_IMAGE_PATH
        image1 = image_path1
        image2 = image_path2

        # perform inference
        results = yolov7.predict([image1, image2], size=1280, augment=True)

        # compare
        self.assertEqual(results.n, 2)
        self.assertEqual(len(results.names), 80)
        self.assertEqual(len(results.pred[0]), 4)
        self.assertEqual(len(results.pred[1]), 5)        
    
    def test_hublike_load_model(self):
        import yolov7

        # init model
        model_path = TestConstants.YOLOV7_MODEL_PATH
        model = yolov7.load(model_path, device=DEVICE)

        # check if loaded
        self.assertNotEqual(model, None)

    def test_hublike_predict(self):
        import yolov7
        from PIL import Image

        # init yolov5s model
        model_path = TestConstants.YOLOV7_MODEL_PATH

        model = yolov7.load(model_path, device=DEVICE)

        # prepare image
        image_path = TestConstants.ZIDANE_IMAGE_PATH
        image = Image.open(image_path)

        # perform inference
        results = model(image, size=640, augment=False)

        # compare
        self.assertEqual(results.n, 1)
        self.assertEqual(len(results.names), 80)
        self.assertEqual(len(results.pred[0]), 4)        
        # init YOLOv7_W6 model
        model_path = TestConstants.YOLOv7_W6_MODEL_PATH
        model = yolov7.load(model_path, device=DEVICE)

        # prepare image
        image_path = TestConstants.BUS_IMAGE_PATH
        image = Image.open(image_path)
        # perform inference
        results = model(image, size=1280, augment=False)

        # compare
        self.assertEqual(results.n, 1)
        self.assertEqual(len(results.names), 80)
        self.assertEqual(len(results.pred[0]), 6)

        # init yolov5s model
        model_path = TestConstants.YOLOV7_MODEL_PATH
        model = yolov7.load(model_path, device=DEVICE)

        # prepare images
        image_path1 = TestConstants.ZIDANE_IMAGE_PATH
        image_path2 = TestConstants.BUS_IMAGE_PATH
        image1 = Image.open(image_path1)
        image2 = Image.open(image_path2)

        # perform inference with multiple images and test augmentation
        results = model([image1, image2], size=1280, augment=True)

        # compare
        self.assertEqual(results.n, 2)
        self.assertEqual(len(results.names), 80)
        self.assertEqual(len(results.pred[0]), 4)
        self.assertEqual(len(results.pred[1]), 5)



if __name__ == "__main__":
    unittest.main()
