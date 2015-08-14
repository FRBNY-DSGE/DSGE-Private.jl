%% Metropolis_Hastings
%%         For nsim*ntimes iterations within each block, generate a new parameter draw.
%%         Decide to accept or reject, and save every ntimes_th draw that is accepted.

function [] = metropolis_hastings(testing, spath, params, mspec, YY, ...
                YY0, YYcoint0, cc0, cc, sigscale, nobs,nlags,nvar,...
                npara,trspec,pmean,pstdd,pshape,TTT_old,RRR_old,...
                CCC_old,valid_old,para_mask,coint,cointadd,cointall,...
                nant, args_nant_antlags, bounds, para_fix, sigpropdim,... 
                sigproplndet,sigpropinv, randvecs, randvals)
  

  
%% Initialize algorithm by drawing para_old from a normal distribution
%% centered on the posterior mode until parameters are within bounds or
%% posterior value is sufficiently large

[TTT,RRR,CCC,valid] = dsgesolv(mspec,params,nant);
retcode = valid;

[lnpost0,lnpy0,zend0,ZZ0,DD0,QQ0] = feval('objfcnmhdsge',params,bounds,YY,YY0,nobs,...
    nlags,nvar,mspec,npara,trspec,pmean,pstdd,pshape,...
    TTT,RRR,CCC,valid,para_mask,coint,cointadd,cointall,YYcoint0,...
    args_nant_antlags{:});

jc = 1;
valid0 = 0;

while ~valid0
  jc = jc+1;

  if testing
    para_old   = params + cc0*(sigscale*randvecs(:, 1));
  else
    para_old   = params + cc0*(sigscale*randn(npara,1));
  end
    
  
  para_old = para_old.*(1-para_mask)+para_fix.*para_mask;
  
  [TTT_old,RRR_old,CCC_old,valid_old] = dsgesolv(mspec,para_old,nant);
  nstate = length(TTT_old);
  nshocks = size(RRR_old,2);

  retcode = valid_old;
  [post_old,like_old,zend_old,ZZ_old,DD_old,QQ_old] = ...
      feval('objfcnmhdsge',para_old,bounds,YY,YY0,nobs,...
      nlags,nvar,mspec,npara,trspec,pmean,pstdd,pshape, ...
      TTT_old,RRR_old,CCC_old,valid_old,para_mask,coint,cointadd,...
      cointall,YYcoint0,args_nant_antlags{:}); 
    
  propdens  = -0.5*sigpropdim*log(2*pi) - 0.5*sigproplndet - 0.5*sigpropdim*log(cc0^2) ...
      -0.5*(para_old - params)'*sigpropinv*(para_old - params)/cc0^2;
  
  if post_old > -10000000;
    valid0 = 1;
  end

  record(jc) = post_old;
  % disp('Initializing Metropolis-Hastings Algorithm')

end

% fprintf(1,'\n Time %3.2f ',ti(I));
% fprintf(1,'\n Peak = %2.3f',lnpost0);


%% Open files for saving
% Parameters draws
outfile1 = [spath,'/params'];
fid1 = fopen(outfile1,'w');

% Transition Matrices
outfile2 = [spath,'/post'];
fid2 = fopen(outfile2,'w');

outfile3 = [spath,'/TTT'];
fid3 = fopen(outfile3,'w');

outfile4 = [spath,'/RRR'];
fid4 = fopen(outfile4,'w');

outfile5 = [spath,'/zend'];
fid5 = fopen(outfile5,'w');


%% Set parameters for number of draws to make/save
nblocks = 22;
nsim = 100;
ntimes = 5;
nburn = nsim*2;



% Initialize variables for calculating rejection rate
Tim = 0;
eT = 0;
reje = 0;



for iblock = 1:nblocks
    
    % if iblock >1; fprintf(1,' block: %2.0f %3.2f ;',[iblock,toc/(iblock-1)]); end;
    
    parasim = zeros(nsim,npara);
    likesim = zeros(nsim,1);
    postsim = zeros(nsim,1);
    rej     = zeros(nsim*ntimes,1);
    TTTsim = zeros(nsim,nstate^2);
    RRRsim = zeros(nsim,nstate*nshocks);
    CCCsim = zeros(nsim,nstate);
    zsim = zeros(nsim,nstate);
    
    
    for j = 1:nsim*ntimes
        
        Tim = Tim+1;
        
        % Draw para_new from the proposal distribution (a multivariate normal
        % distribution centered on the previous draw, with standard
        % deviation cc*sigscale).
        
        if testing
            para_new = para_old + cc*(sigscale*randvecs(:, j));
        else
            para_new = para_old + cc*(sigscale*randn(npara,1));
        end
        
        para_new = para_new.*(1-para_mask)+para_fix.*para_mask;
        [TTT_new,RRR_new,CCC_new,valid_new] = dsgesolv(mspec,para_new,nant);
        retcode = valid_new;
        
        % Solve the model, check that parameters are within bounds, and
        % evalue the posterior.
        
        [post_new,like_new,zend_new,ZZ_new,DD_new,QQ_new] = feval('objfcnmhdsge',para_new,bounds,YY,YY0,nobs,...
            nlags,nvar,mspec,npara,trspec,pmean,pstdd,pshape,TTT_new,RRR_new,CCC_new,valid_new,para_mask,coint,cointadd,cointall,YYcoint0,args_nant_antlags{:});
        
        fprintf(1,'Iteration %2.0f: posterior = %4.8f\n',[j, post_new]);

        
        % Calculate the multivariate log likelihood of jump from para_old to para_new
        propdens = -0.5*sigpropdim*log(2*pi) - 0.5*sigproplndet - 0.5*sigpropdim*log(cc^2) ...
            -0.5*(para_new - para_old)'*sigpropinv*(para_new - para_old)/cc^2;
        
        
        % Choose to accept or reject the new parameter by calculating the
        % ratio (r) of the new posterior value relative to the old one
        % We compare min(1,r) to a number drawn randomly from a
        % uniform (0,1) distribution. This allows us to always accept
        % the new draw if its posterior value is greater than the previous draw's,
        % but it gives some probability to accepting a draw with a smaller posterior value,
        % so that we may explore tails and other local modes.
        
        r = min([1 ; exp( post_new - post_old)]);
        
        if testing
            x = randvals(j);
        else
            x = rand(1, 1);
        end
        
        if x < r;
            % Accept proposed jump
            para_old = para_new;
            post_old = post_new;
            like_old = like_new;
            
            TTT_old = TTT_new;
            RRR_old = RRR_new;
            CCC_old = CCC_new;
            valid_old = valid_new;
            
            zend_old = zend_new;
            ZZ_old = ZZ_new;
            DD_old = DD_new;
            QQ_old = QQ_new;
            
            % fprintf(1,'Iteration %2.0f: accept proposed jump\n',j);
        else
            % Reject proposed jump
            rej(j) = 1;
            reje = reje+1;
            
            % fprintf(1,'Iteration %2.0f: reject proposed jump\n',j);
        end
        
        % Save every (ntime)th draw
        if (j/ntimes) == round(j/ntimes)
            likesim(j/ntimes,1) = like_old;
            postsim(j/ntimes,1) = post_old;
            parasim(j/ntimes,:) = para_old';
            TTTsim(j/ntimes,:) = TTT_old(:)';
            RRRsim(j/ntimes,:) = RRR_old(:)';
            CCCsim(j/ntimes,:) = CCC_old(:)';
            zsim(j/ntimes,:) = zend_old(:)';
        end
        
        if j == nsim*ntimes
             fprintf(1,'\nRejection perct %2.4f ',[sum(rej)/j]);
        end
        
    end
    
    fwrite(fid1,parasim','single');
    fwrite(fid2,postsim','single');
    fwrite(fid3,TTTsim','single');
    fwrite(fid4,RRRsim','single');
    fwrite(fid5,zsim','single');


end

fclose(fid1);
fclose(fid2);
fclose(fid3);
fclose(fid4);
fclose(fid5);



% fprintf(1,'rejection rate = %2.4f',reje/Tim);