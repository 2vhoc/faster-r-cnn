# Faster R-CNN Tutorial Implementation AsusTUF Gaming F15 i5 11400H RTX 2050

## Giới thiệu
Faster R-CNN là một mô hình deep learning phổ biến trong bài toán Object Detection (phát hiện vật thể). Repo này cung cấp một implementation đơn giản của Faster R-CNN để giúp người mới học hiểu rõ về cách hoạt động của mô hình.

## Cấu trúc của Faster R-CNN
Mô hình Faster R-CNN bao gồm 3 thành phần chính:

1. **Backbone Network (CNN)**
   - Sử dụng VGG16 làm backbone
   - Trích xuất feature maps từ ảnh đầu vào
   - Output là feature map có kích thước HxWxD

2. **Region Proposal Network (RPN)**
   - Tạo ra các anchor boxes
   - Dự đoán objectness score cho mỗi anchor
   - Đề xuất các vùng có khả năng chứa object (region proposals)

3. **Detection Network**
   - ROI Pooling để chuẩn hóa kích thước feature maps
   - Phân loại đối tượng trong mỗi proposal
   - Tinh chỉnh bounding box

## Cài đặt

### Yêu cầu
```
python 3.7+
pytorch 1.7+
torchvision
numpy
opencv-python
matplotlib
```

### Cài đặt các thư viện
```bash
pip install -r requirements.txt
```

## Cấu trúc thư mục
```
faster-rcnn/
│
├── data/                # Thư mục chứa dataset
├── src/
│   ├── model.py        # Định nghĩa mô hình Faster R-CNN
│   ├── rpn.py          # Region Proposal Network
│   ├── detector.py     # Detection Network
│   ├── utils.py        # Các hàm tiện ích
│   └── train.py        # Mã nguồn training
│
└── README.md
```

## Hướng dẫn sử dụng

### 1. Chuẩn bị dữ liệu
```python
# Cấu trúc thư mục data
data/
  ├── images/           # Chứa ảnh training
  └── annotations/      # Chứa file annotation dạng txt hoặc xml
```

### 2. Training
```python
# Chạy training
python src/train.py --data_path data/ --epochs 10
```

### 3. Dự đoán
```python
# Dự đoán trên ảnh mới
python src/predict.py --image path/to/image.jpg --model models/model.pth
```

## Giải thích code

### 1. Region Proposal Network (RPN)
```python
class RPN(nn.Module):
    def __init__(self):
        super(RPN, self).__init__()
        # Conv layer chung
        self.conv = nn.Conv2d(512, 512, 3, 1, 1)
        # Layer dự đoán objectness score
        self.cls_layer = nn.Conv2d(512, 2*9, 1, 1, 0) 
        # Layer dự đoán box coordinates
        self.bbox_layer = nn.Conv2d(512, 4*9, 1, 1, 0)
```

### 2. ROI Pooling
```python
def roi_pool(feature_map, rois, output_size):
    """
    Args:
        feature_map: Features từ backbone
        rois: Region proposals từ RPN
        output_size: Kích thước output mong muốn
    Returns:
        pooled_features: Features đã được chuẩn hóa kích thước
    """
    # Code implementation
```

## Kết quả
- Training loss qua các epoch
- Ví dụ kết quả dự đoán trên một số ảnh test
- Độ chính xác trên tập validation

## Tips khi training
1. **Chuẩn bị dữ liệu**:
   - Đảm bảo annotation chính xác
   - Cân bằng số lượng các class
   - Augmentation data đơn giản

2. **Hyperparameters**:
   - Learning rate: 0.001
   - Batch size: 2-4 (tùy vào GPU)
   - Anchor scales: [8, 16, 32]
   - IoU thresholds: 0.7 (positive), 0.3 (negative)

3. **Training**:
   - Train backbone trước
   - Sau đó fine-tune toàn bộ mạng
   - Lưu lại model có val_loss tốt nhất

## Tài liệu tham khảo
1. Paper gốc: [Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks](https://arxiv.org/abs/1506.01497)
2. [Tutorial PyTorch Object Detection](https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html)

## Hỗ trợ
Nếu bạn có bất kỳ câu hỏi nào, vui lòng:
- Tạo issue trong repo
- Email: 2vhoc7@gmail.com
