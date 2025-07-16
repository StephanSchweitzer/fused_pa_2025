#!/bin/bash
# Docker build test script

echo "Testing Docker build..."

# Build the Docker image
docker build -t ml-pipeline-test .

if [ $? -eq 0 ]; then
    echo "✓ Docker build successful"
else
    echo "✗ Docker build failed"
    exit 1
fi

# Test that the image can run basic commands
echo "Testing basic container functionality..."
docker run --rm ml-pipeline-test python --version

if [ $? -eq 0 ]; then
    echo "✓ Container can run Python"
else
    echo "✗ Container failed to run Python"
    exit 1
fi

# Test that pipeline modules can be imported
echo "Testing pipeline imports in container..."
docker run --rm ml-pipeline-test python -c "from pipeline_utils import PipelineLogger; print('✓ Pipeline imports work')"

if [ $? -eq 0 ]; then
    echo "✓ Pipeline modules can be imported in container"
else
    echo "✗ Pipeline modules failed to import in container"
    exit 1
fi

echo "🎉 All Docker tests passed!"