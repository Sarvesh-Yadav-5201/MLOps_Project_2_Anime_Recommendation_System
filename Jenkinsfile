pipeline {
    agent any

    stages {
        stage ('Cloning from Github ......'){
            steps{
                script{
                    echo 'Cloning from Github ......'
                    
                    checkout scmGit(branches: [[name: '*/master']], extensions: [], userRemoteConfigs: [[credentialsId: 'github_token', url: 'https://github.com/Sarvesh-Yadav-5201/MLOps_Project_2_Anime_Recommendation_System.git']])
                }
            }
        }
    }

}