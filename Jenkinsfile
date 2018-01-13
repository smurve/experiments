pipeline {
  agent {
    docker {
        image 'ubuntu:16.04'
    }
  }
  stages {
    stage('identify') {
      steps {
        sh 'whoami'
      }
    }
    stage('locate') {
      steps {
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
