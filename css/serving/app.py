# Copyright (c) 2023, Salesforce, Inc.
# SPDX-License-Identifier: Apache-2
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import annotations

import os
import time
import uuid

from comet_mpm import CometMPM
from flask import Flask, request
import pandas as pd

from css.serving.decoder import decode
from css.serving.encoder import encode
from css.serving.model_cache import ModelCache

app = Flask(__name__)


@app.get("/ping")
def ping():
    """Health check endpoint to ensure the service is operational."""
    return {"message": "ok"}


def _log_predictions(dataframe, input_data):
    mpm = CometMPM(
        workspace_name="collin-cunningham",
        model_name="css-model",
        model_version="1.0.0",
        api_key=os.env.get('API_KEY'),
    )
    for (_, row), (_, input_row) in zip(
        dataframe.iterrows(), input_data.iterrows(), strict=False
    ):
        input_features = row.xs("metric", level=2)
        input_features.index = [f"{c[0]}__{c[1]}" for c in input_features.index]
        output_features = row.xs("metric", level=2)
        output_features.index = [f"{c[0]}__{c[1]}" for c in output_features.index]
        input_features_dict = input_features.to_dict()
        input_features_dict["salesforce__acct_age"] = input_row["acct_age"]
        input_features_dict["salesforce__acct_size"] = input_row["acct_size"]
        output_features_dict = output_features.to_dict()
        mpm.log_event(
            prediction_id=str(uuid.uuid4()),
            input_features=input_features_dict,
            output_value=0,
            output_features=output_features_dict,
            timestamp=time.time(),
        )
    mpm.end()


@app.post("/invocations")
def invocations():
    """Model invoke endpoint."""
    input_data = decode(request.data, request.content_type)
    model = ModelCache.model()
    if set(model.min_required_columns).issubset(input_data.columns):
        predictions = model.score(input_data)
        _log_predictions(predictions, input_data)
    elif len(model.min_required_columns) == input_data.shape[1]:
        input_data.columns = model.min_required_columns
        predictions = model.score(input_data)
        _log_predictions(predictions, input_data)
    else:
        message = (
            f"Model requires columns {model.min_required_columns}, "
            f"but received columns {input_data.columns}. If you pass unlabeled "
            "data, it must match exactly the dimensionality and ordering "
            f"of `min_required_columns`."
        )
        return {"message": message}, 400

    predictions["last_modified"] = pd.Timestamp.now().isoformat()
    return encode(predictions, request.accept_mimetypes)


def start_server():
    print("Starting Server...")
    app.run(host="0.0.0.0", port=8080)
