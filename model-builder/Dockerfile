# TensorFlow is far too large for a zip Lambda, so the converter ships as a container image.
# BUILD CONTEXT = the app root (apps/ai-on-edges-demo) so we can also copy web/architectures.json.
# The CDK stack (infra/) sets this context automatically.
FROM public.ecr.aws/lambda/python:3.12

COPY model-builder/requirements-lambda.txt ./
RUN pip install --no-cache-dir -r requirements-lambda.txt

COPY model-builder/converter.py model-builder/architectures.py model-builder/lambda_function.py ${LAMBDA_TASK_ROOT}/
# Co-locate the spec so architectures.py finds it inside the container (there is no ../web here).
COPY web/architectures.json ${LAMBDA_TASK_ROOT}/architectures.json

CMD ["lambda_function.handler"]
