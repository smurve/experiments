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
          // available as an env variable, but will be masked if you try to print it out any which way
          sh 'echo $PASSWORD'
          // also available as a Groovy variable—note double quotes for string interpolation
          echo "$USERNAME"
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
