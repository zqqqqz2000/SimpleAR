import json
from typing import Dict, List

import t2v_metrics
import torch


class FineVQAReward:
    def __init__(self, model="clip-flant5-xxl", device="cuda", log_file="./log_testcurr.jsonl"):
        self.model = t2v_metrics.VQAScore(model=model, cache_dir="./reward_model/clip-flant5-xxl")
        self.device = device
        self.reward_model = self.model.to(self.device)
        self.reward_model.eval()

        self.log_file = log_file
        with open(self.log_file, "w", encoding="utf-8") as f:
            pass

    def __call__(self, images: List[str], scenes: List[Dict]):
        vqa_score_list = []
        for image, scene in zip(images, scenes):
            questions = []
            dependencies = []
            question_types = []

            # Process each category
            categories = ["object", "count", "attribute", "relation"]
            for category in categories:
                for item in scene["qa"][category]:
                    questions.append(item["question"])
                    dependencies.append(item["dependencies"])
                    question_types.append(category)

            support_data = {"questions": questions, "dependencies": dependencies, "question_types": question_types}

            vqa_score = self.reward_model([image], support_data["questions"])[0]
            vqa_score = vqa_score.tolist()
            sum_score = 0
            for score, dependency, question_type in zip(vqa_score, dependencies, question_types):
                if question_type == "object":
                    sum_score += score
                elif question_type == "attribute" or question_type == "count":
                    try:
                        sum_score += score * vqa_score[dependency[0] - 1]
                    except IndexError as e:
                        print(f"vqascore:{vqa_score}, type:{question_types}, dependency:{dependency}")
                elif question_type == "relation":
                    try:
                        sum_score += score * (min(vqa_score[dependency[0] - 1], vqa_score[dependency[1] - 1]))
                    except IndexError as e:
                        print(f"vqascore:{vqa_score}, type:{question_types}, dependency:{dependency}")
                else:
                    raise ValueError("Not implemented question type error")

            avg_vqa_score = sum_score / len(questions)  # assume questions is not an empty list
            vqa_score_list.append(avg_vqa_score)
            log_data = {
                "image_path": image,
                "difficulty": scene["difficulty"],
                "prompt": scene["prompt"],
                "questions": questions,
                "vqa_score": vqa_score,
                "avg_vqa_score": avg_vqa_score,
            }

            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(log_data, ensure_ascii=False) + "\n")

        return torch.tensor(vqa_score_list, dtype=torch.float32, device=self.device)
