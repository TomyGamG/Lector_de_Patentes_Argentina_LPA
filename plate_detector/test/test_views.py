import pytest
from django.urls import reverse
from django.core.files.uploadedfile import SimpleUploadedFile

@pytest.mark.django_db
def test_upload_plate_view_auth(client, django_user_model):

    # crear usuario
    user = django_user_model.objects.create_user(username="test", password="1234")
    client.login(username="test", password="1234")

    img = SimpleUploadedFile("test.jpg", b"fake", content_type="image/jpeg")
    
    url = reverse("analyze_plate")
    response = client.post(url, {"image": img})

    assert response.status_code == 200
    data = response.json()
    assert "status" in data
