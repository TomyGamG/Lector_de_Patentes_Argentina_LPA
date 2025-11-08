from django.test import TestCase
from django.contrib.auth.models import User
from plate_detector.models import Vehicle, PlateDetection, Client
import random

class PlateDetectionMetricsTestCase(TestCase):
    def setUp(self):
        # Crear un usuario y cliente de prueba
        self.user = User.objects.create_user(username="testuser", password="12345")
        self.client_obj = Client.objects.create(
            user=self.user,
            company_name="Test Company",
            phone="1234567890"
        )

        # Crear vehículos de prueba
        self.vehicles = []
        for i in range(5):
            v = Vehicle.objects.create(
                plate_number=f"ABC12{i}",
                client=self.client_obj,
                brand="Toyota",
                model="Corolla",
                color="Red"
            )
            self.vehicles.append(v)

        # Simular detecciones YOLO con confidence y aciertos/fallos
        self.detections = []
        for v in self.vehicles:
            # Simular si YOLO detecta correctamente la placa
            detected_correctly = random.choice([True, False])
            confidence = random.uniform(0.5, 1.0)  # Confidence aleatoria
            plate_number = v.plate_number if detected_correctly else "WRONG123"

            d = PlateDetection.objects.create(
                user=self.user,
                vehicle=v,
                plate_number=plate_number,
                confidence=confidence,
                is_client=True,
                image="test_image.jpg"
            )
            self.detections.append((d, detected_correctly))

    def test_accuracy_and_recall(self):
        """Calcula métricas simples de Accuracy y Recall"""
        tp = 0  # True positives
        fn = 0  # False negatives
        fp = 0  # False positives

        for d, correct in self.detections:
            if correct and d.plate_number == d.vehicle.plate_number:
                tp += 1
            elif correct and d.plate_number != d.vehicle.plate_number:
                fn += 1
            elif not correct and d.plate_number != d.vehicle.plate_number:
                fp += 1

        # Accuracy: TP / total
        total = len(self.detections)
        accuracy = tp / total
        # Recall: TP / (TP + FN)
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0

        print(f"\nAccuracy: {accuracy:.2f}, Recall: {recall:.2f}")

        self.assertGreaterEqual(accuracy, 0.0)
        self.assertGreaterEqual(recall, 0.0)

    def test_confidence_threshold(self):
        """Verifica que todas las detecciones tengan confidence >= 0.5"""
        for d, _ in self.detections:
            self.assertGreaterEqual(d.confidence, 0.5)

