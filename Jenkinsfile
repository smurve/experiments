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
    stage('virtualenv') {
      steps {
        sh 'source init_env.sh venv'
      }
    }
    stage('unit test') {
      steps {
        sh 'pytest'
      }
    }
    stage('build') {
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
    stage('hello') {
      steps {
        echo 'Hello from within the container (hopefully...)'
      }
    }
  }
}
