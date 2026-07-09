#!/usr/bin/env python3
"""AWS CDK app: deploy the converter as a container Lambda behind an HTTPS Function URL.

    cd infra && pip install -r requirements.txt && cdk deploy

Outputs `ConverterUrl` — set it as the browser's ?converter=<url> (or CONVERTER_URL) to
convert in the cloud instead of via the local MQTT builder. Function URL auth is NONE.
"""
import os

import aws_cdk as cdk
from aws_cdk import Stack, Duration, CfnOutput
from aws_cdk import aws_lambda as _lambda
from constructs import Construct

# App root is the parent of this infra/ dir — used as the Docker build context so the image
# can COPY both model-builder/*.py and web/architectures.json.
APP_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class ConverterStack(Stack):
    def __init__(self, scope: Construct, cid: str, **kw):
        super().__init__(scope, cid, **kw)

        fn = _lambda.DockerImageFunction(
            self,
            "Converter",
            code=_lambda.DockerImageCode.from_image_asset(
                directory=APP_ROOT,
                file="model-builder/Dockerfile",
            ),
            memory_size=2048,  # TensorFlow needs headroom; more memory also = more CPU
            timeout=Duration.seconds(60),
        )

        url = fn.add_function_url(
            auth_type=_lambda.FunctionUrlAuthType.NONE,  # no auth, per requirements
            cors=_lambda.FunctionUrlCorsOptions(
                allowed_origins=["*"],
                allowed_methods=[_lambda.HttpMethod.ALL],
                allowed_headers=["*"],
            ),
        )
        CfnOutput(self, "ConverterUrl", value=url.url)


app = cdk.App()
ConverterStack(app, "AiOnEdgesConverter")
app.synth()
