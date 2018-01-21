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
        withCredentials([usernamePassword(credentialsId: 'dockerhub', usernameVariable: 'USERNAME', passwordVariable: 'PASSWORD')]) {
          sh 'docker build -t smurve/capsnet-fashion-trainer:latest -f Dockerfile-mnist-trainer-gpu .'
          sh 'docker login --password $PASSWORD --username $USERNAME'
          sh 'docker push smurve/capsnet-fashion-trainer:latest'
        }
      }
    }
    stage('start trainer job') {
      steps {
        sh './start_training_job.sh'
      }
    }
    stage('build inference') {
      steps {
        withCredentials([usernamePassword(credentialsId: 'dockerhub', usernameVariable: 'USERNAME', passwordVariable: 'PASSWORD')]) {
          sh 'docker build -t smurve/capsnet-fashion:latest .'
          sh 'docker login --password $PASSWORD --username $USERNAME'
          sh 'docker push smurve/capsnet-fashion:latest'
        }
      } 
    }
    stage('system test') {
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
