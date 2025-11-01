# Razer Hydra - AI-Powered Image-to-Drawing System

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-green.svg)](https://opencv.org)

**RAZER HYDRA** is a state-of-the-art system that transforms any image into a hand-drawn sketch using advanced AI models and precise hardware control. The system processes images through multiple stages—including Real-ESRGAN super-resolution, AI-based sketch generation, and intelligent path optimization—before translating them into smooth pen strokes on a digital canvas using Razer device hardware.

![Project Overview](https://via.placeholder.com/800x400?text=AI+Image+to+Drawing+Pipeline)

## ✨ Features

### 🤖 **Advanced AI Processing Pipeline**

- **Real-ESRGAN Super-Resolution**: 4x upscaling for crisp, detailed input images
- **Anime2Sketch AI Model**: State-of-the-art edge extraction and line art generation
- **Adaptive Contrast Enhancement**: CLAHE-based preprocessing for enhanced line detection
- **Intelligent Post-Processing**: Multi-stage noise reduction and cleanup algorithms

### 🎯 **Precision Hardware Control**

- **Direct Razer Device Integration**: Low-level hardware control for absolute precision
- **Dynamic Path Optimization**: Smart stroke ordering and path stitching algorithms
- **Real-time Performance Monitoring**: Live progress tracking with detailed analytics
- **Emergency Stop Functionality**: Instant halt capability with ESC key

### 🎨 **Sophisticated Drawing Engine**

- **Multi-Grid Path Planning**: Optimized stroke ordering for efficient drawing
- **Smooth Interpolation**: Pixel-perfect line rendering with configurable smoothness
- **Batch Command Processing**: High-speed execution with API call optimization
- **Canvas Calibration**: Precise coordinate mapping for any drawing surface

### 📊 **Professional User Interface**

- **Rich Console Interface**: Real-time status monitoring with live progress bars
- **Interactive Calibration**: Guided setup process with audio feedback
- **Detailed Analytics**: Performance metrics, ETA calculations, and drawing statistics
- **Live Stroke Preview**: Braille-based canvas preview during drawing process

## 🏗️ Technical Architecture

### **Processing Stages**

```
Input Image → Real-ESRGAN Upscaling → AI Sketch Generation → Post-Processing → Path Optimization → Hardware Drawing
```

#### **Stage 0: Real-ESRGAN Super-Resolution**

The system employs [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) for high-quality image upscaling:

- **Model**: `realesrgan-x4plus-anime` (optimized for anime/artistic content)
- **Enhancement**: 4x resolution improvement before AI processing
- **Performance**: Background processing with seamless integration
- **Fallback**: Graceful degradation if upscaler unavailable

#### **Stage 1: AI Art Generation**

Powered by the [Anime2Sketch](https://github.com/Mukosame/Anime2Sketch) model:

- **Architecture**: Deep learning model specialized in edge extraction
- **Input Processing**: Adaptive square padding and normalization
- **CLAHE Enhancement**: Contrast-limited adaptive histogram equalization
- **Output**: High-quality grayscale line art with enhanced edge definition

#### **Stage 2: Post-Processing & Optimization**

Advanced image processing pipeline:

- **Median Blur**: Noise reduction while preserving edge integrity
- **Global Thresholding**: Binary conversion with configurable sensitivity
- **Connected Component Analysis**: Removal of small artifacts and noise
- **Pixel-level Cleanup**: Configurable minimum area filtering

#### **Stage 3: Intelligent Path Planning**

Sophisticated optimization algorithms:

- **Segment Stitching**: Automatic connection of adjacent line segments
- **Grid-based Ordering**: Spatial organization for efficient stroke execution
- **Nearest Neighbor Sorting**: Minimized pen lift movements
- **Trajectory Optimization**: Smooth interpolation with configurable precision

## 🚀 Quick Start

### **Prerequisites**

```bash
# Python 3.8+ required
python --version

# Install dependencies
pip install -r requirements.txt
```

### **Required Hardware**

- **Razer Device**: Compatible Razer mouse with Synapse driver
- **Drawing Surface**: MS Paint or similar application for canvas
- **Administrator Privileges**: Required for hardware access

### **Model Setup**

Download the required AI models:

```bash
# Create models directory
mkdir -p weights upscaler

# Download Real-ESRGAN (place in upscaler/ directory)
# https://github.com/xinntao/Real-ESRGAN/releases

# Download Anime2Sketch weights (place in weights/ directory)
# https://github.com/Mukosame/Anime2Sketch/releases
```

### **Running the System**

```bash
# Launch with administrator privileges
python main.py
```

### **User Workflow**

1. **Image Selection**: Choose any image file (JPG, PNG, BMP)
2. **Canvas Calibration**: Set drawing area boundaries in MS Paint
3. **AI Processing**: Watch real-time processing with live status
4. **Drawing Execution**: Monitor progress with detailed analytics
5. **Completion**: Review results and choose next action

## ⚙️ Configuration

The system provides extensive configuration options in `main.py`:

### **Upscaling Parameters**

```python
USE_UPSCALER = True              # Enable Real-ESRGAN processing
UPSCALER_MODEL = 'realesrgan-x4plus-anime'  # Model selection
```

### **AI Processing Settings**

```python
MODEL_TYPE = 'default'           # 'default' or 'improved'
LOAD_SIZE = 2048                 # Processing resolution (1024-4096)
CLAHE_CLIP_LIMIT = 4.0           # Contrast enhancement (0 to disable)
```

### **Post-Processing Options**

```python
USE_POST_PROCESSING = True       # Enable cleanup algorithms
MEDIAN_BLUR_SIZE = 3            # Noise reduction strength
BINARY_THRESHOLD = 240          # Line detection sensitivity
MINIMUM_PIXEL_AREA = 40         # Artifact removal threshold
```

### **Drawing Performance**

```python
DRAWING_SPEED_PERCENT = 100     # Overall speed multiplier
GLIDE_STEP_SIZE = 4             # Pen movement smoothness
BATCH_SIZE = 20                 # Command batch size
INTER_STROKE_DELAY_SEC = 0.001  # Base delay between strokes
```

## 📈 Performance Metrics

The system provides comprehensive performance monitoring:

### **Real-time Statistics**

- **Drawing Progress**: Live percentage and stroke count
- **Time Estimation**: Dynamic ETA calculation
- **API Performance**: Command rate and data throughput
- **Stroke Analytics**: Complexity and execution timing
- **Hardware Utilization**: Device interaction monitoring

### **Quality Metrics**

- **Processing Stages**: Visual debugging output in `debug_output/`
- **Line Quality**: Configurable threshold and cleanup parameters
- **Path Efficiency**: Optimized stroke ordering statistics
- **Hardware Precision**: Coordinate mapping accuracy

## 🛠️ Technical Implementation

### **Hardware Integration**

- **Windows API**: Direct device communication via SetupAPI
- **IOCTL Commands**: Low-level mouse control protocol
- **Real-time Timing**: High-precision sleep with Windows multimedia timer
- **Error Handling**: Robust device detection and recovery

### **AI Model Architecture**

- **PyTorch Backend**: CUDA/DirectML GPU acceleration support
- **Model Loading**: Dynamic device detection and optimization
- **Tensor Processing**: Efficient image transformation pipeline
- **Memory Management**: Optimized tensor operations

### **Image Processing Pipeline**

- **OpenCV Integration**: Advanced computer vision algorithms
- **PIL Compatibility**: Seamless image format handling
- **NumPy Operations**: High-performance numerical processing
- **Multi-stage Processing**: Modular algorithm composition

## 🔧 Advanced Features

### **Emergency Controls**

- **ESC Key**: Immediate drawing halt at any time
- **Audio Feedback**: Sound notifications for all major events
- **Visual Indicators**: Real-time pen state and status display
- **Graceful Shutdown**: Clean resource management and cleanup

### **Debug and Diagnostics**

- **Debug Output**: Complete processing stage visualization
- **Performance Logging**: Detailed timing and statistics
- **Error Recovery**: Robust exception handling and fallback modes
- **Development Tools**: Interactive parameter adjustment

### **Customization Options**

- **Model Selection**: Multiple AI model support
- **Parameter Tuning**: Extensive configuration flexibility
- **UI Themes**: Customizable interface appearance
- **Sound Integration**: Personalized audio feedback

## 📁 Project Structure

```
project/
├── main.py                 # Main application entry point
├── model.py               # AI model definitions
├── requirements.txt       # Python dependencies
├── weights/              # AI model weights
├── upscaler/             # Real-ESRGAN executable
├── sounds/               # Audio feedback files
└── debug_output/         # Processing stage outputs
```

## 🎯 Use Cases

### **Creative Applications**

- **Art Reproduction**: Convert photos to line art for traditional drawing
- **Educational Tools**: Interactive art creation and learning
- **Prototype Development**: Rapid concept visualization
- **Therapeutic Applications**: Art therapy and rehabilitation

### **Technical Applications**

- **CAD Conversion**: Technical drawing digitization
- **Pattern Generation**: Template and design creation
- **Quality Assurance**: Visual inspection and verification
- **Research Applications**: Computer vision algorithm testing

### **Development Areas**

- **AI Model Integration**: Support for additional edge detection models
- **Hardware Compatibility**: Extended device support beyond Razer
- **Platform Expansion**: Cross-platform compatibility development
- **Performance Optimization**: Enhanced speed and efficiency algorithms

## 🙏 Acknowledgments

- **Real-ESRGAN Team**: For the exceptional super-resolution technology
- **Anime2Sketch Authors**: For pioneering AI-based sketch generation
- **PyTorch Community**: For the robust deep learning framework
- **OpenCV Contributors**: For comprehensive computer vision tools
- **Razer Developer**: For hardware integration documentation

---

**RAZER HYDRA** - Where artificial intelligence meets precision artistry. Transform any image into a masterpiece with the power of AI and hardware control.

*Built with ❤️ using Python, PyTorch, OpenCV, and advanced hardware integration.*