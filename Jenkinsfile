pipeline {
  agent {
    docker {
        image 'ubuntu:16.04'
    }
  }
  stages {
    stage('test') {
      steps {
        sh 'uname -a'
        sh 'whoami'
        echo 'Hello from within the container (hopefully...)'
      }
    }
  }
}
