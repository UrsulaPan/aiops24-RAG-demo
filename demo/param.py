from llama_index.core import (

    VectorStoreIndex,

    load_index_from_storage,

    StorageContext,

    ServiceContext,

)
from llama_index.experimental.param_tuner import RayTuneParamTuner

from llama_index.experimental.param_tuner.base import  RunResult

from llama_index.core.evaluation.eval_utils import (
    get_responses,
    aget_responses,
)
from llama_index.core.evaluation import (
    SemanticSimilarityEvaluator,
    BatchEvalRunner,
)
from llama_index.legacy.llms import OpenAILike as OpenAI
from llama_index.core.evaluation import QueryResponseDataset
import os

import numpy as np

from pathlib import Path
from dotenv import dotenv_values
from pipeline.ingestion import read_data
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.core.schema import IndexNode
docs = read_data("/mnt/workspace/aiops24-RAG-demo/demo/data")

param_dict = {"chunk_size": [256, 512, 1024], "top_k": [1, 2, 5]}

eval_dataset = QueryResponseDataset.from_json(

    "/mnt/workspace/aiops24-RAG-demo/demo/data/llama2_eval_qr_dataset.json"

)
eval_qs = eval_dataset.questions
print(eval_qs[:1])

ref_response_strs = [r for (_, r) in eval_dataset.qr_pairs]

fixed_param_dict = {

    "docs": docs,

    "eval_qs": eval_qs[:10],

    "ref_response_strs": ref_response_strs[:10],

}

def _build_index(chunk_size):


    index_out_path = f"./storage_{chunk_size}"

    if not os.path.exists(index_out_path):

        Path(index_out_path).mkdir(parents=True, exist_ok=True)

        # parse docs

        node_parser = SimpleNodeParser.from_defaults(chunk_size=chunk_size)

        base_nodes = node_parser.get_nodes_from_documents(docs)

        # build index

        index = VectorStoreIndex(base_nodes)

        # save index to disk

        index.storage_context.persist(index_out_path)

    else:

        # rebuild storage context

        storage_context = StorageContext.from_defaults(

            persist_dir=index_out_path

        )

        # load index

        index = load_index_from_storage(
            storage_context,
        )
    return index

def _get_eval_batch_runner():
    config = dotenv_values(".env")
    llm = OpenAI(
            api_key=config["GLM_KEY"],
            model="glm-4",
            api_base="https://open.bigmodel.cn/api/paas/v4/",
            is_chat_model=True,
        )

    eval_service_context = ServiceContext.from_defaults(

        llm=llm

    )

    evaluator_s = SemanticSimilarityEvaluator(

        service_context=eval_service_context

    )

    eval_batch_runner = BatchEvalRunner(

        {"semantic_similarity": evaluator_s}, workers=2, show_progress=True

    )

    

    return eval_batch_runner

async def objective_function(params_dict):
    chunk_size = params_dict["chunk_size"]
    docs = params_dict["docs"]
    top_k = params_dict["top_k"]
    eval_qs = params_dict["eval_qs"]
    ref_response_strs = params_dict["ref_response_strs"]
    # build index
    index = _build_index(chunk_size, docs)
    # query engine
    query_engine = index.as_query_engine(similarity_top_k=top_k)
    # get predicted responses
    pred_response_objs = await aget_responses(
        eval_qs, query_engine, show_progress=True
    )
    # run evaluator
    eval_batch_runner = _get_eval_batch_runner()
    eval_results = await eval_batch_runner.aevaluate_responses(
        eval_qs, responses=pred_response_objs, reference=ref_response_strs
    )
    # get semantic similarity metric
    mean_score = np.array(
        [r.score for r in eval_results["semantic_similarity"]]
    ).mean()

    return RunResult(score=mean_score, params=params_dict)

param_tuner = RayTuneParamTuner(
    param_fn=objective_function,
    param_dict=param_dict,
    fixed_param_dict=fixed_param_dict,
    run_config_dict={"storage_path": "/tmp/custom/ray_tune", "name": "my_exp"},
)

results = param_tuner.tune()
best_result = results.best_run_result

best_top_k = results.best_run_result.params["top_k"]

best_chunk_size = results.best_run_result.params["chunk_size"]

print(f"Score: {best_result.score}")

print(f"Top-k: {best_top_k}")

print(f"Chunk size: {best_chunk_size}")





