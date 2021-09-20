from adr_detection.transformer_based_arc import ModelSpans
from remove_intersections import CreateTwoDataset
from create_iob import CreateIOBPredictions
import os
from neg_detection.negex import NegExScope
from neg_detection.bertneg import BERTneg
from config import *
from utils.utils import save_results, get_folder_name, get_evaluated_raw_dataset,mic
from tqdm import tqdm

folder_name = get_folder_name()

for model in tqdm(Config.MODELS_TO_EVALUATE.value, total=len(Config.MODELS_TO_EVALUATE.value)):
    mic(model)
    for dataset in Config.DATASETS_TO_EVALUATE.value:
        mic(dataset)
        dataset_to_evaluate = f"{model}.pickle"
        evaluation_path = os.path.join(Config.PATH_INPUT.value, Config.MODEL_PREDS.value, dataset_to_evaluate)
        mic(evaluation_path)

        # original dataset with columns tweet_id and tweet (text)
        evaluated_raw_dataset = get_evaluated_raw_dataset(
            path = os.path.join(Config.PATH_INPUT.value, Config.TO_EVALUATE_PATH.value, dataset+".pickle")
        )
        try:
            df = ModelSpans().evaluate(
                evaluation_path, 
                task=Config.TASK.value,
                original_dataset = evaluated_raw_dataset,
                no_gold_labels = False)
            mic(df)
            mic("ADE detection results imported!")
        except:
            mic(evaluation_path)
            break
        if Config.NEG_MODEL.value == "bertneg":
            scope_builder = BERTneg(
                train = not os.path.exists(os.path.join(Config.PATH_OUTPUT.value, "models", "bertneg_bioscope.model"))
            )   
            df = scope_builder.predict(df)
        if Config.NEG_MODEL.value == "negex":
            scope_builder = NegExScope(
                df = df.copy(),
                task = Config.TASK.value,
                build = True,
                save_intermediate = True
            )   
            df = scope_builder.run_negex()
            if "negation_scope_iob" in df.columns:
                df = df.drop(columns=['negation_scope_iob'])

        mic("Negation scope detection terminated!")
        
        df, df_baseline = CreateTwoDataset(df).create(keep_intersection = False)
        df = CreateIOBPredictions(df).create_iob()
        df_baseline = CreateIOBPredictions(df_baseline).create_iob()
        mic("IOB encoding terminated!")
        save_results(df,df_baseline,model,dataset_to_evaluate,folder_name)
        mic(f"Iteration terminated!")