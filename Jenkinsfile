pipeline {
  agent {label 'GTX1080'}
  stages {
    stage('identify and locate') {
      steps {
        sh 'id'
        sh 'uname -a'
        sh 'pwd'
        sh 'ls'
      }
    }
    stage('unit test') {
      steps {
        sh 'rm -rf venv'
        sh '. ./shell/init_env && cd src && pytest'
      }
    }
    stage('build trainer') {
      steps {
        /*
            Note that you need to have dockerhub credentials in Jenkins for the docker push command to succeed.
        */
        withCredentials([usernamePassword(credentialsId: 'dockerhub', usernameVariable: 'USERNAME', passwordVariable: 'PASSWORD')]) {
          sh 'docker build -t smurve/capsnet-fashion-trainer:latest-gpu -f Dockerfile-mnist-trainer-gpu .'
          sh 'docker login --password $PASSWORD --username $USERNAME'
          sh 'docker push smurve/capsnet-fashion-trainer:latest-gpu'
        }
      }
    }
    stage('start trainer job') {
      steps {
        sh './shell/start_training_job.sh'
      }
    }
    stage('build inference') {
      steps {
        withCredentials([usernamePassword(credentialsId: 'dockerhub', usernameVariable: 'USERNAME', passwordVariable: 'PASSWORD')]) {
          sh 'docker build -t smurve/ellie_inference_webapp:latest -f Dockerfile-inference-webapp .'
          sh 'docker login --password $PASSWORD --username $USERNAME'
          sh 'docker push smurve/ellie_inference_webapp:latest'
        }
      } 
    }
    stage('inference health check') {
      steps {
        sh './shell/run_webapp_health.sh'
      }
    }
    stage('deploy inference service') {
      steps {
        sh 'kubectl delete -f k8s/inference || echo inference service did not exist. Fine.'
        sh 'kubectl create -f k8s/inference'
      }
    }
  }
}
