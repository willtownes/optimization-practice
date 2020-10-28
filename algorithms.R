#Optimization algorithms

MAXITER<-2000
TOLERANCE<-1e-6

l2norm<-function(x){sqrt(sum(x^2))}

############## First Order ##############

grad_descent<-function(f,g,x_init,stp=.001,maxIter=MAXITER){
  delta<-100
  t<-1
  res<-matrix(0,nrow=maxIter,ncol=3)
  colnames(res)<-c("x","y","f")
  res[1,]<-c(x_init,f(x_init))
  while(abs(delta)>TOLERANCE && t<maxIter){
    t<-t+1
    grad<-g(res[t-1,1:2])
    res[t,1:2]<-res[t-1,1:2]-stp*grad
    res[t,3]<-f(res[t,1:2])
    delta<-res[t,3]-res[t-1,3]
  }
  as.data.frame(res[1:t,])
}

momentum<-function(f,g,x_init,stp=.001,maxIter=MAXITER,momentum=.9,wait=1){
  delta<-100
  t<-1
  res<-matrix(0,nrow=maxIter,ncol=3)
  colnames(res)<-c("x","y","f")
  res[1,]<-c(x_init,f(x_init))
  velo<-0
  while(abs(delta)>TOLERANCE && t<maxIter){
    t<-t+1
    if(t<wait) velo<-0
    velo<-momentum*velo
    grad<-g(res[t-1,1:2])
    velo<-velo+stp*grad
    res[t,1:2]<-res[t-1,1:2]-velo
    res[t,3]<-f(res[t,1:2])
    delta<-res[t,3]-res[t-1,3]
  }
  as.data.frame(res[1:t,])
}

nesterov<-function(f,g,x_init,stp=.001,maxIter=MAXITER,momentum=.9,wait=1){
  delta<-100
  t<-1
  res<-matrix(0,nrow=maxIter,ncol=3)
  colnames(res)<-c("x","y","f")
  res[1,]<-c(x_init,f(x_init))
  velo<-0
  while(abs(delta)>TOLERANCE && t<maxIter){
    t<-t+1
    if(t<wait) velo<-0
    velo<-momentum*velo
    grad<-g(res[t-1,1:2]-velo)
    velo<-velo+stp*grad
    res[t,1:2]<-res[t-1,1:2]-velo
    res[t,3]<-f(res[t,1:2])
    delta<-res[t,3]-res[t-1,3]
  }
  as.data.frame(res[1:t,])
}

irprop<-function(f,g,x_init,maxIter=MAXITER,eta_plus=1.2,eta_minus=.5,
                 delta_max=50,delta_min=0,delta0=.5){
  k<-length(x_init)
  chg<-100
  t<-1
  res<-matrix(0,nrow=maxIter,ncol=3)
  colnames(res)<-c("x","y","f")
  res[1,]<-c(x_init,f(x_init))
  delta_old<-delta<-delta0*rep(1,k)
  dw<-grad_old<-rep(0,k)
  while(abs(chg)>TOLERANCE && t<maxIter){
    #while(t<maxIter){
    grad<-g(res[t,1:k])
    ind<-sign(grad_old)*sign(grad)
    delta[ind>0]<-min(delta_old[ind>0]*eta_plus, delta_max)
    delta[ind<0]<-max(delta_old[ind<0]*eta_minus, delta_min)
    grad[ind<0]<-1e-4*grad[ind<0] #comment out this line to get "original" rprop
    
    #irprop+
    #dw[ind>=0]<- -sign(grad[ind>=0])*delta[ind>=0]
    #dw[ind<0]<- if(chg>0) -dw[ind<0] else 0
    #irprop-
    dw<- -sign(grad)*delta
    
    res[t+1,1:k]<-res[t,1:k]+dw
    res[t+1,k+1]<-f(res[t+1,1:k])
    chg<-res[t+1,k+1]-res[t,k+1]
    delta_old<-delta
    grad_old<-grad
    t<-t+1
  }
  as.data.frame(res[1:t,])
}

adamax<-function(f,g,x_init,maxIter=MAXITER,stp=.002,
                 beta1=.9,beta2=.999){
  k<-length(x_init)
  chg<-100
  t<-1
  res<-matrix(0,nrow=maxIter,ncol=3)
  colnames(res)<-c("x","y","f")
  res[1,]<-c(x_init,f(x_init))
  m<-rep(0,k)
  u<-rep(0,k)
  while(abs(chg)>TOLERANCE && t<maxIter){
    t<-t+1
    grad<-g(res[t-1,1:k])
    m<-beta1*m+(1-beta1)*grad
    u<-pmax(beta2*u,abs(grad))
    velo<- -(stp/(1-beta1^t))*m/u
    res[t,1:k]<-res[t-1,1:k]+velo
    res[t,k+1]<-f(res[t,1:k])
    chg<-res[t,k+1]-res[t-1,k+1]
  }
  as.data.frame(res[1:t,])
}

nadam<-function(f,g,x_init,maxIter=MAXITER,stp=.002,
                beta1=.9,beta2=.999){
  k<-length(x_init)
  chg<-100
  t<-1
  res<-matrix(0,nrow=maxIter,ncol=3)
  colnames(res)<-c("x","y","f")
  res[1,]<-c(x_init,f(x_init))
  m<-rep(0,k)
  u<-rep(0,k)
  while(abs(chg)>TOLERANCE && t<maxIter){
    t<-t+1
    grad<-g(res[t-1,1:k])
    m<-beta1*m+(1-beta1)*grad
    u<-pmax(beta2*u,abs(grad))
    velo<- -(stp/u)*(beta1*m+grad*(1-beta1))/(1-beta1^t)
    res[t,1:k]<-res[t-1,1:k]+velo
    res[t,k+1]<-f(res[t,1:k])
    chg<-res[t,k+1]-res[t-1,k+1]
  }
  as.data.frame(res[1:t,])
}

avagrad<-function(f,g,x_init,maxIter=MAXITER,stp=.1,
                  beta1=.9,beta2=.999,adam_eps_param=.1){
  #Savarese et al 2019 algorithm 2
  #stp=alpha, adam_eps_param=epsilon
  #defaults copied from https://github.com/lolemacs/avagrad/blob/master/optimizers.py
  k<-length(x_init) #Savarese calls this "d"
  chg<-100
  t<-1
  res<-matrix(0,nrow=maxIter,ncol=3)
  colnames(res)<-c("x","y","f")
  res[1,]<-c(x_init,f(x_init))
  m<-rep(0,k)
  v<-rep(0,k)
  while(abs(chg)>TOLERANCE && t<maxIter){
    t<-t+1
    grad<-g(res[t-1,1:k])
    m<-beta1*m+(1-beta1)*grad
    eta<-1/(adam_eps_param+sqrt(v))
    res[t,1:k]<-res[t-1,1:k] - stp*(eta/l2norm(eta/sqrt(k)))*m
    v<-beta2*v+(1-beta2)*grad^2
    res[t,k+1]<-f(res[t,1:k])
    chg<-res[t,k+1]-res[t-1,k+1]
  }
  as.data.frame(res[1:t,])
}

nloptr_lbfgs<-function(f,g,x_init,maxIter=MAXITER){
  opts<-list(algorithm="NLOPT_LD_LBFGS",ftol_abs=TOLERANCE,maxeval=maxIter)
  res<-matrix(0,nrow=maxIter,ncol=3)
  colnames(res)<-c("x","y","f")
  t<-0
  ff<-function(x){
    t<<-t+1
    res[t,1:2]<<-x
    return(res[t,3]<<-f(x))
  }
  gg<-function(x){ g(x) }
  fit<-nloptr::nloptr(x0=x_init,eval_f=ff,eval_grad_f=gg,opts=opts)
  if(fit$status<0){
    warning("NLOPT failed to converge with status: ",fit$status)
  } else if(fit$status==5){
    warning("NLOPT reached max iteration limit without converging!")
  } else {
    message("NLOPT converged with status: ",fit$status)
  }
  unique(as.data.frame(res[1:t,]))
}

############## First Order ##############

newton_full<-function(f,g,h,x_init,maxIter=MAXITER){
  k<-length(x_init)
  chg<-100
  t<-1
  res<-matrix(0,nrow=maxIter,ncol=3)
  colnames(res)<-c("x","y","f")
  res[1,]<-c(x_init,f(x_init))
  while(abs(chg)>TOLERANCE && t<maxIter){
    #message("t=",t,"chg=",signif(chg,3))
    t<-t+1
    grad<-g(res[t-1,1:k])
    hess<-h(res[t-1,1:k])
    res[t,1:k]<-res[t-1,1:k] - solve(hess,grad)
    res[t,k+1]<-f(res[t,1:k])
    chg<-res[t,k+1]-res[t-1,k+1]
  }
  as.data.frame(res[1:t,])
}

newton_diag<-function(f,g,h,x_init,maxIter=MAXITER,stp=.1){
  k<-length(x_init)
  chg<-100
  t<-1
  res<-matrix(0,nrow=maxIter,ncol=3)
  colnames(res)<-c("x","y","f")
  res[1,]<-c(x_init,f(x_init))
  while(abs(chg)>TOLERANCE && t<maxIter){
    t<-t+1
    grad<-g(res[t-1,1:k])
    hess<-h(res[t-1,1:k])
    res[t,1:k]<-res[t-1,1:k] - stp*grad/diag(hess)
    res[t,k+1]<-f(res[t,1:k])
    chg<-res[t,k+1]-res[t-1,k+1]
  }
  as.data.frame(res[1:t,])
}
