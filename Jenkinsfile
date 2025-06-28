pipeline {
    agent any

    environment {
        // Define environment variables here
        VENV_DIR = 'venv'
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
                withCredentials([string(credentialsId: 'gcp_key_anime', variable: 'GOOGLE_APPLICATION_CREDENTIALS')]) {
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
    }

}