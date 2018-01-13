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
    stage('build') {
      steps {
        withCredentials([usernamePassword(credentialsId: 'dockerhub', usernameVariable: 'USERNAME', passwordVariable: 'PASSWORD')]) {

          sh 'docker build -t smurve/capsnet-fashion:test .
          sh 'docker login --password $PASSWORD --username $USERNAME'
          sh 'docker push'
        }
      } 
    }
    stage('hello') {
      steps {
        echo 'Hello from within the container (hopefully...)'
      }
    }
  }
}
