pipeline {
  agent {
    docker {
        image 'ubuntu:16.04'
    }
  }
  stages {
    stage('test') {
      steps {
        echo 'Hello from $(uname -a)'
      }
    }
  }
}
