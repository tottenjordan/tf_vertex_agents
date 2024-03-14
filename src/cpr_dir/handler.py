
import json
import logging
from fastapi import Response
from google.cloud.aiplatform.prediction.handler import PredictionHandler

class CprHandler(PredictionHandler):
    """
    Default prediction handler for the pred requests sent to the application
    """

    async def handle(self, request):
        """Handles a prediction request."""
        
        request_body = await request.body()
        logging.info(f'request_body: {request_body}')
        
        request_body_dict = json.loads(request_body)
        logging.info(f'request_body_dict: {request_body_dict}')
        
        instances=request_body_dict["instances"]
        logging.info(f'instances: {instances}')
        
        prediction_results = self._predictor.postprocess(
            self._predictor.predict(
                self._predictor.preprocess(instances)
            )
        )
                                                         
        logging.info(f'prediction: {prediction_results}')

        return Response(content=json.dumps(prediction_results))
