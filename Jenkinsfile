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
        sh '. ./init_env.sh && pytest'
      }
    }
    stage('build trainer') {
      steps {
        withCredentials([usernamePassword(credentialsId: 'dockerhub', usernameVariable: 'USERNAME', passwordVariable: 'PASSWORD')]) {
          sh 'docker build -t smurve/capsnet-fashion-trainer:latest .'
          sh 'docker login --password $PASSWORD --username $USERNAME'
          sh 'docker push smurve/capsnet-fashion-trainer:latest'
        }
      }
    }
    stage('start trainer job') {
      steps {
        'sh ./start_training_job.sh'
      }
    }
    stage('build inference') {
      steps {
        withCredentials([usernamePassword(credentialsId: 'dockerhub', usernameVariable: 'USERNAME', passwordVariable: 'PASSWORD')]) {
          sh 'docker build -t smurve/capsnet-fashion:test .'
          sh 'docker login --password $PASSWORD --username $USERNAME'
          sh 'docker push smurve/capsnet-fashion:test'
        }
      } 
    }
    stage('system test') {
      steps {
        sh './runtest.sh'
      }
    }
  }
}
