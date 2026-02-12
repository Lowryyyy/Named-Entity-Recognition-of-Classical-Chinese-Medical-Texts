# Named-Entity-Recognition-of-Classical-Chinese-Medical-Texts
# 训练模型
python run_ner_with_knowledge.py train \
    --data_dir ./data/tcm_ner \
    --output_dir ./models/tcm_ner_model \
    --model_name bert-base-chinese \
    --epochs 10 \
    --batch_size 16 \
    --learning_rate 2e-5

# 预测
python run_ner_with_knowledge.py predict \
    --model_dir ./models/tcm_ner_model \
    --text "太阳病，发热而渴，不恶寒者，为温病" \
    --output_file ./results/predictions.txt

# 批量预测
python run_ner_with_knowledge.py predict \
    --model_dir ./models/tcm_ner_model \
    --text_file ./data/test_texts.txt \
    --output_file ./results/batch_predictions.txt
