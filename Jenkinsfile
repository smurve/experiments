pipeline {
  agent {
    docker {
        image 'ubuntu:16.04'
    }
  }
  stages {
    stage('test') {
      steps {
        uname -a
        whoami
        echo 'Hello from within the container (hopefully...)'
      }
    }
  }
}
