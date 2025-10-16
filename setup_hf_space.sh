#!/bin/bash

# Setup script for HuggingFace Spaces deployment

echo "🚀 Setting up HuggingFace Space deployment..."

# Create deployment directory
DEPLOY_DIR="hf_space_deploy"
mkdir -p "$DEPLOY_DIR"

echo "📁 Creating deployment directory: $DEPLOY_DIR"

# Copy main application
cp demo/app.py "$DEPLOY_DIR/"
cp demo/requirements.txt "$DEPLOY_DIR/requirements.txt"

# Copy demo data
if [ -d "demo/demo_curated" ]; then
    cp -r demo/demo_curated "$DEPLOY_DIR/"
    echo "✅ Copied demo_curated/"
fi

# Copy additional examples
if [ -f "demo/scatter_temp_energy.png" ]; then
    cp demo/scatter_temp_energy.png "$DEPLOY_DIR/"
    echo "✅ Copied scatter_temp_energy.png"
fi

# Create README for HF Space
cat > "$DEPLOY_DIR/README.md" << 'EOL'
---
title: AskAnything in Charts - Qwen 2.5
emoji: 📊
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 4.0.0
app_file: app.py
pinned: false
license: mit
python_version: 3.10
---

# AskAnything in Charts - Powered by Qwen 2.5

Compare Qwen 2.5 7B base model with fine-tuned version for chart understanding!

## 🎯 Performance
- **Qwen 2.5 7B:** 57.5% accuracy
- **Qwen 2.5 7B + LORA:** 60.0% accuracy
- **Improvement:** +2.5% absolute

## 🎨 Features
- Side-by-side comparison
- 8 curated examples showing improvements
- Upload your own charts!

Built with ❤️ using Qwen2.5-VL and HuggingFace Transformers
EOL

echo "✅ Created HF Space README"

echo ""
echo "🎉 HuggingFace Space setup complete!"
echo "📦 Files ready in: $DEPLOY_DIR/"
echo ""
echo "📋 Next steps:"
echo "1. Upload your adapter to HuggingFace Hub:"
echo "   huggingface-cli upload prakashchhipa/Qwen2.5-VL-7B-ChartQA-LoRA outputs/qwen2_5_vl_7b_sft_chart_text_oct15_e2/"
echo "2. Update ADAPTER_PATH in $DEPLOY_DIR/app.py to: prakashchhipa/Qwen2.5-VL-7B-ChartQA-LoRA"
echo "3. Create a new Space on HuggingFace"
echo "4. Upload files from $DEPLOY_DIR/"
