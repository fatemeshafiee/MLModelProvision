# MLModelProvision
This repository contains an implementation of the ML Model Provision Service, which is part of the Network Data Analytics Function (NWDAF). Each subscription request to this service should specify the kind of ML model it wants by providing a list of parameters, including its use case (such as slice load level or UE abnormal behaviour), in addition to other features.
Based on the request, this service will serve the ML model and provide the URL for inference to the subscriber. To store the ML models and manage their life cycle, this implementation integrates the service with MLflow.


# Build the image
```bash
docker build -t <dockerhub-username>/<repo-name>:<tag> . 
```
# Push the image 

```bash 
docker push <dockerhub-username>/<repo-name>:<tag>
```

if using our [`open5gs-k8s-nwdaf`](https://github.com/fatemeshafiee/open5gs-k8s-nwdaf.git), `<dockerhub-username>/<repo-name>:<tag>` should be used as an image in the [`mlflow-deployment.yaml`] (https://github.com/fatemeshafiee/open5gs-k8s-nwdaf/blob/main/NWDAF/MLflow/mlflow-deployment.yaml) 

Admin can store ML models and log, register, and serve them in MLflow to be available for ML Model Provision Service subscribers by exec-ing into the container and using MLflow directly. We also implemented an admin API that can receive the ML model from outside the container by providing the link that the model should be downloaded from, and the description of the ML model. An example of using this API is provided in `log_model_v2.sh`.



These repositories provide the reference implementation and artifacts for the following paper. If you use this code or build on it, please cite:

Fatemeh Shafiei Ardestani, Niloy Saha, Noura Limam, and Raouf Boutaba, “Towards NWDAF-enabled Analytics and Closed-Loop Automation in 5G Networks”, arXiv:2505.06789, 2025.
