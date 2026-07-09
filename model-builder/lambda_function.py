"""AWS Lambda handler for the converter (container image, wired to a Function URL).

Same contract as server.py: POST {arch_id, weights} -> .tflite bytes. Function URL auth is
NONE (no security, per requirements). Binary is returned base64+isBase64Encoded, which the
Function URL decodes back to raw bytes for the browser.
"""
import base64
import json

import converter

CORS = {
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Methods": "POST, OPTIONS",
    "Access-Control-Allow-Headers": "Content-Type",
    "Access-Control-Expose-Headers": "X-Convert-Ms, X-Max-Abs-Err",
}


def handler(event, context):
    method = event.get("requestContext", {}).get("http", {}).get("method", "POST")
    if method == "OPTIONS":
        return {"statusCode": 204, "headers": CORS, "body": ""}
    try:
        body = event.get("body", "") or ""
        if event.get("isBase64Encoded"):
            body = base64.b64decode(body).decode()
        payload = json.loads(body)
        tflite, meta = converter.convert_weights(
            payload["arch_id"], payload["weights"],
            quantize=payload.get("quantize", False), freqs=payload.get("freqs"))
        headers = dict(CORS)
        headers["Content-Type"] = "application/octet-stream"
        headers["X-Convert-Ms"] = str(meta["convert_ms"])
        headers["X-Max-Abs-Err"] = str(meta["max_abs_err"])
        return {
            "statusCode": 200,
            "headers": headers,
            "body": base64.b64encode(tflite).decode(),
            "isBase64Encoded": True,
        }
    except Exception as e:
        return {
            "statusCode": 400,
            "headers": {**CORS, "Content-Type": "application/json"},
            "body": json.dumps({"error": str(e)}),
        }
