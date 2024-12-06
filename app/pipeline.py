from clearml import PipelineController
from loguru import logger

logger.info("Pipeline started")

pipe = PipelineController(
    name="CS-GY-6613-RAG-Pipeline",
    project="cs-gy-6613-rag",
    version="1.0.0",
    add_pipeline_tags=False,
)

pipe.set_default_execution_queue("default")

pipe.add_step(
    name="test", base_task_project="cs-gy-6613-rag", base_task_name="Pipeline Task 1"
)
# pipe.add_step(
#     name="test1",
#     parents=[
#         "test",
#     ],
#     base_task_project="cs-gy-6613-rag",
#     base_task_name="Pipeline Task 2",
# )
# pipe.add_step(
#     name="test2",
#     parents=[
#         "test1",
#     ],
#     base_task_project="cs-gy-6613-rag",
#     base_task_name="Pipeline Task 3",
# )

pipe.start(queue="default")

logger.info("Pipeline done")
