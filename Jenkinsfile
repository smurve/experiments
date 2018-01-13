pipeline {
  agent {
    docker {
        image 'ubuntu:16.04'
    }
  }
  stages {
    stage('identify and locate') {
      steps {
        sh 'id'
        sh 'uname -a'
      }
    }
    stage('hello') {
      steps {
        echo 'Hello from within the container (hopefully...)'
      }
    }
  }
}
