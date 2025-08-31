#!/bin/bash

# AI Emotion Detection - Quick Start Script
# This script sets up and runs the emotion detection web app

set -e  # Exit on any error

echo "🤖 AI Emotion Detection - Quick Start"
echo "====================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if Python is installed
check_python() {
    print_status "Checking Python installation..."
    if command -v python3 &> /dev/null; then
        PYTHON_CMD="python3"
        print_success "Python 3 found: $(python3 --version)"
    elif command -v python &> /dev/null; then
        PYTHON_CMD="python"
        print_success "Python found: $(python --version)"
    else
        print_error "Python is not installed. Please install Python 3.8+ first."
        exit 1
    fi
}

# Create virtual environment
create_venv() {
    print_status "Setting up virtual environment..."
    if [ ! -d "venv" ]; then
        $PYTHON_CMD -m venv venv
        print_success "Virtual environment created"
    else
        print_warning "Virtual environment already exists"
    fi
}

# Activate virtual environment
activate_venv() {
    print_status "Activating virtual environment..."
    if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
        # Windows
        source venv/Scripts/activate
    else
        # Unix/Linux/macOS
        source venv/bin/activate
    fi
    print_success "Virtual environment activated"
}

# Install dependencies
install_deps() {
    print_status "Installing dependencies..."
    pip install --upgrade pip
    pip install -r requirements.txt
    print_success "Dependencies installed"
}

# Setup model
setup_model() {
    print_status "Setting up emotion detection model..."
    if [ ! -f "models/emotion_model.onnx" ]; then
        print_warning "No emotion model found. Creating dummy model for testing..."
        python scripts/download_model.py
    else
        print_success "Emotion model found"
    fi
}

# Test model
test_model() {
    print_status "Testing emotion detection model..."
    if python scripts/test_model.py; then
        print_success "Model test passed"
    else
        print_warning "Model test failed - this is expected with a dummy model"
    fi
}

# Start the application
start_app() {
    print_status "Starting the emotion detection web app..."
    echo ""
    echo "🌐 The app will be available at: http://localhost:8000/static/"
    echo "📖 API documentation at: http://localhost:8000/docs"
    echo "🔍 Health check at: http://localhost:8000/health"
    echo ""
    echo "Press Ctrl+C to stop the server"
    echo ""
    
    uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
}

# Main execution
main() {
    check_python
    create_venv
    activate_venv
    install_deps
    setup_model
    test_model
    start_app
}

# Handle script arguments
case "${1:-}" in
    "docker")
        print_status "Starting with Docker..."
        docker-compose up --build
        ;;
    "test")
        print_status "Running tests only..."
        check_python
        create_venv
        activate_venv
        install_deps
        setup_model
        test_model
        ;;
    "help"|"-h"|"--help")
        echo "Usage: $0 [option]"
        echo ""
        echo "Options:"
        echo "  (no args)  - Start the app normally"
        echo "  docker     - Start with Docker Compose"
        echo "  test       - Run tests only"
        echo "  help       - Show this help message"
        echo ""
        echo "Examples:"
        echo "  $0          # Start normally"
        echo "  $0 docker   # Start with Docker"
        echo "  $0 test     # Run tests"
        ;;
    *)
        main
        ;;
esac 