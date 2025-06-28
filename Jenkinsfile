pipeline {
    agent any

    environment {
        // Define environment variables here
        VENV_DIR = 'venv'

        // More Environment Variables 
        // 1. GCP PROJECT :  Get from the Google Cloud Console 
        GCP_PROJECT = 'single-arcadia-463020-t4'
        // 2. GCLOUD_PATH :  Path to the gcloud SDK installation
        GCLOUD_PATH = '/var/jenkins_home/google-cloud-sdk/bin'
        // 3. KUBECONFIG :  Path to the kubeconfig file for GKE
        KUBECTL_AUTH_PLUGIN = '/usr/lib/google-cloud-sdk/bin'

    }

    stages {

        // Stage to clone the repository from GitHub
        stage ('Cloning from Github ......'){
            steps{
                script{
                    echo 'Cloning from Github ......'
                    
                    checkout scmGit(branches: [[name: '*/master']], extensions: [], userRemoteConfigs: [[credentialsId: 'github_token', url: 'https://github.com/Sarvesh-Yadav-5201/MLOps_Project_2_Anime_Recommendation_System.git']])
                }
            }
        }

        // Stage to create a virtual environment
        stage ('Making a Virtual Environment inside Jenkins ......'){
            steps{
                script{
                    echo 'Making a Virtual Environment inside Jenkins ......'
                    
                    sh '''
                    python3 -m venv $VENV_DIR
                    . ${VENV_DIR}/bin/activate
                    pip install --upgrade pip
                    pip install -e .
                    pip install dvc
                    '''
                }
            }
        }

        // Stage to pull data using DVC
        stage ('DVC PULL ......'){
            steps{
                withCredentials([file(credentialsId: 'gcp_key_anime', variable: 'GOOGLE_APPLICATION_CREDENTIALS')]) {
                    script {
                        echo 'Pulling data using DVC ......'
                        
                        sh '''
                        . ${VENV_DIR}/bin/activate
                        dvc pull
                        '''
                    }
                }
                
            }
        }

        // Stage to build and push the Docker image to Google Container Registry (GCR)
        stage ('BUILD AND PUSH IMAGE TO GCR ......'){
            steps{
                withCredentials([file(credentialsId: 'gcp_key_anime', variable: 'GOOGLE_APPLICATION_CREDENTIALS')]) {
                    script {
                        echo 'BUILD AND PUSH IMAGE TO GCR ......'
                        
                        sh '''
                        export PATH=${PATH}:${GCLOUD_PATH} 
                        gcloud auth activate-service-account --key-file=${GOOGLE_APPLICATION_CREDENTIALS}
                        gcloud config set project ${GCP_PROJECT}
                        gcloud auth configure-docker --quiet
                        docker build -t gcr.io/${GCP_PROJECT}/anime-recommender:latest .
                        docker push gcr.io/${GCP_PROJECT}/anime-recommender:latest
                        '''
                    }
                }
                
            }
        }

        // Stage to deploy the application to Kubernetes
        stage ('Deploying to Kubernetes ......'){
            steps{
                withCredentials([file(credentialsId: 'gcp_key_anime', variable: 'GOOGLE_APPLICATION_CREDENTIALS')]) {
                    script {
                        echo 'Deploying to Kubernetes  ......'
                        
                        sh '''
                        export PATH=${PATH}:${GCLOUD_PATH}:${KUBECTL_AUTH_PLUGIN}
                        gcloud auth activate-service-account --key-file=${GOOGLE_APPLICATION_CREDENTIALS}
                        gcloud config set project ${GCP_PROJECT}

                        gcloud container clusters get-credentials anime-recommender-cluster --region us-central1
                        kubectl apply -f deployment.yaml

                        '''
                    }
                }
                
            }
        }
    }

}