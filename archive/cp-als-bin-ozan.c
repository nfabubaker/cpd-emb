#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdarg.h>
#include <unistd.h>
#include <mpi.h>
#include <math.h>
#include <time.h>
#include <limits.h>

#include <chrono>

#include "mkl.h"
#include "mkl_spblas.h"

#define min(x,y) (((x) < (y)) ? (x) : (y))
#define WARMUP 5
#define TOTTIME 3.0

struct tensor
{
  int mype;
  int npes;
	
  int gnnz;      // number of nonzeros in the overall tensor
  int nmodes;    // number of modes of the tensor 
  int *gdims;    // array of size nmodes, gdims[i] denotes the size in mode i+1 
  int *ldims;    // local dimensions
  int **indmap;  // relabeling of indices local to the processor

  int **gpart;   // array of pointers, gpart[i] points to the partition array of rows in mode i+1 
  // for cyclic dist: gpart[i][j] denotes the owner processor of row j of mode i, for checkerboard dist: gpart[i][j] denoted the mesh layer of row j 

  int **interpart;   //ckbd distribution, mesh layer id
  int **intrapart;   //ckbd distribution, owner of the row inside mesh layer


  int meshsize;  // overall number of processors
  int *meshdims; // array of size nmodes, meshdims[i] denotes the number of processors in mode i in virtual processor mesh 
  int *meshinds; // the indices of the processor at each mode

  int *chunksize; // array of size nmodes, denoting the size of the chunk along each mode IN TERMS OF PARTS

  int *inds;
  double *vals;

  int *sporder;    //sparsity order
  int nnz;         //total number of nonzeros of my subtensor

  //cartesian nd processors 
  MPI_Comm *layercomm;
  int *layermype;
  int *layersize;
	
  struct comm *comm;


  // factor matrices
  int cprank;
  double **mat;

  // small dense U^T.U matrices for each factor matrix U
  double *uTu;

  // fiber storage?
  int fiber;

  // all-to-all communication?
  int alltoall;

  // checkerboard
  int ckbd;
};


struct fibertensor
{
  int lmode;

  int *fibermode;  //the modes of the fibers; nmodes-1 same, one different
  int **order;     //order of the modes from top to the bottom for modes
  int *topmostcnt; //number of nonzero subtensors for the topmost mode
  int ***xfibers;  //hierarchical pointer structure, pointing in the bottommost level either to lfibers or slfibers

  int *lfibers;    // fibers of the longest mode
  int *slfibers;    // fibers of the second longest mode

  double *lvals;   //values of the fibers of the longest mode
  double *slvals;  //values of the fibers of the second longest modep
};

struct comm
{
  int *nrecvwho;
  int **recvwho;
  int **xrecvind; // number of factor matrix rows to be received from each comunicated processor at each mode
  // e.g. in a 2X3x4 mesh, processor 0 is going to communicate with 0 2 4 .. 22 for first mode. nrecv[0][2] for proc 0 denotes the amount communicated with proc 4. pairs of (pid and number)
  int **recvind;

  int *nsendwho;
  int **sendwho;
  int **xsendind;
  int **sendind;

  double *buffer;

};

struct stat
{
  int *recvvol;
  int *sendvol;

  int *recvmsg;
  int *sendmsg;

  int *row;
  int nnz;

};

void *myMalloc(size_t size)
{
  void *p = malloc(size);
  if(p == NULL)
    {
      printf("myMalloc couldn't allocate %d byte memory\n", size);
      exit(1);
    }
  else
    return p;
	
}

void setintzero(int *arr, int size)
{
  int i;
  for(i = 0; i < size; i++)
    arr[i] = 0;
}

void setdoublezero(double *arr, int size)
{
  int i;
  for(i = 0; i < size; i++)
    arr[i] = 0.0;
}



void free_tensor(struct tensor *t);


long get_wc_time ( void )
{
	return 0;//OZANOZAN
  /*static struct timeval twclk ;
  gettimeofday(&twclk, NULL) ;
  return(twclk.tv_sec*1000000 + twclk.tv_usec) ;
  */
}


void get_chunk_info(struct tensor *t)
{
  int i, j, nchunks, nmodes, meshsize, mult, *meshdims, *chunksize;

  nchunks = -1;
  nmodes = t->nmodes;
  meshdims = t->meshdims;
  meshsize = t->meshsize;

  // compute number of chunks per processor
  for(i = 0; i < nmodes; i++)
    if(meshdims[i] != 1)
      {	if(nchunks == -1)
          nchunks = 1;
        else
          for(j = i; j < nmodes; j++)
            nchunks *= meshdims[j];
      }

  if(meshsize == 1)
    nchunks = 1;

  // compute chunk size in terms of parts
  t->chunksize = (int *)myMalloc(sizeof(int)*nmodes);
  chunksize = t->chunksize;

  mult = 1;
  for(i = 0; i < nmodes; i++)
    {
      if(meshdims[i] == 1)
        chunksize[i] = meshsize;
      else
        {
          chunksize[i] = mult;
          mult *= meshdims[i];
        }
    }

}

void read_cyclic_tensor_nonzeros(char *tfile, struct tensor *t)
{
  int i, j, p, pind, cs, *part, dim, meshdim, mult, nmodes, *gdims, *meshinds, *meshdims, **own, gnnz, nnz;

  mult = t->mype;
  nmodes = t->nmodes;
  gdims = t->gdims;
  meshdims = t->meshdims;
  meshinds = t->meshinds;

  // find the org ids that I own nonzeros in
  own = (int **)myMalloc(sizeof(int *)*nmodes);
  for(i = 0; i < nmodes; i++)
    {	
      part = t->gpart[i];
      dim = t->gdims[i];
      meshdim = meshdims[i];
      pind = meshinds[i];
      cs = t->chunksize[i];

      own[i] = (int *)myMalloc(sizeof(int)*dim);		
      for(j = 0; j < dim; j++)
        {
          own[i][j] = 0;
          p = part[j];
          if((p / cs) % meshdim == pind)
            own[i][j] = 1;
        }
    }
 

  char line[1024], *str;
  int ptr, indx, mine;


  int nzid = 0;
  gnnz = t->gnnz;
  int *mynnzs = (int *)myMalloc(sizeof(int)*gnnz);
  setintzero(mynnzs, gnnz);

  nnz = 0;
  // just count nonzeros at my portion
  FILE *tf = fopen(tfile, "r");
  fgets(line, 1024, tf);
  while(fgets(line, 1024, tf) != NULL)
    {
      ptr = 0;
      mine = 0;
      for(i = 0; i < nmodes; i++ && str != NULL)
        {
          str = strtok (&line[ptr], "\t");
          ptr += strlen(str) + 1;
          indx =  atoi(str)-1;
			
          if(own[i][indx])
            mine++;
          else
            break;
        }
		
      if(mine == nmodes)
        {
          nnz++;
          mynnzs[nzid] = 1;
        }
      nzid++;
    }

  for(i = 0 ; i < nmodes; i++)
    free(own[i]);
  free(own);


  t->inds = (int *)myMalloc(nmodes*nnz*sizeof(int));
  t->vals = (double *)myMalloc(nnz*sizeof(double));
  t->nnz = nnz;
	
  // now I am going to get the nonzeros in my portion
  rewind(tf);
  fgets(line, 1024, tf);
  nzid = 0;
  nnz = 0;
  int indptr = 0;
  while(fgets(line, 1024, tf) != NULL)
    {
      if(mynnzs[nzid])
        {
          ptr = 0;
          for(i = 0; i < nmodes; i++ && str != NULL)
            {
              str = strtok (&line[ptr], "\t");
              ptr += strlen(str) + 1;
              t->inds[indptr++] = atoi(str)-1;;

            }
          str = strtok (&line[ptr], "\t");
          t->vals[nnz++] = atof(str);
        }
      nzid++;
    }

  fclose(tf);
  free(mynnzs);

}

void read_cyclic_tensor_nonzeros_rootreads(char *tfile, struct tensor *t)
{
  int *nnzs, mynnz, *disp, i, who, indx, ptr,  nmodes, meshind, mult, **gpart, *chunksize, *meshdims, *allinds;
  double *allvals;
  char line[1024], *str;
  FILE *tf;

  nmodes = t->nmodes;

  if(t->mype == 0)
    {
      gpart = t->gpart;
      chunksize = t->chunksize;
      meshdims = t->meshdims;

      nnzs = (int *)myMalloc(t->npes*sizeof(int));
      setintzero(nnzs, t->npes);

      tf = fopen(tfile, "r");
      fgets(line, 1024, tf);
      while(fgets(line, 1024, tf) != NULL)
        {
          mult = 1;
          who = 0;
          ptr = 0;
          for(i = 0; i < nmodes; i++ && str != NULL)
            {
              str = strtok (&line[ptr], "\t");
              ptr += strlen(str) + 1;
              indx =  atoi(str)-1;
				
              meshind = (gpart[i][indx] / chunksize[i]) % meshdims[i];
              who += meshind*mult;
              mult *= meshdims[i];
            }
          if(who > t->npes)
            printf("yanlis hesapladin %d\n", who);
          nnzs[who]++;
        }
      fclose(tf);
    }

  MPI_Scatter(nnzs, 1, MPI_INT, &mynnz, 1, MPI_INT, 0, MPI_COMM_WORLD);

  t->nnz = mynnz;
  t->inds = (int *)myMalloc(mynnz*nmodes*sizeof(int));
  t->vals = (double *)myMalloc(mynnz*sizeof(double));
	
  if(t->mype == 0)
    {
      int *singlennz = (int*) myMalloc(nmodes*sizeof(int));

      disp = (int *)myMalloc((t->npes+2)*sizeof(int));
      setintzero(disp, t->npes+2);
		
      for(i = 2; i < t->npes+2; i++)
        disp[i] += disp[i-1] + nnzs[i-2];

      if(disp[t->npes+1] != t->gnnz)
        printf("error\n");

      allinds = (int *)myMalloc(t->gnnz*nmodes*sizeof(int));
      allvals = (double *)myMalloc(t->gnnz*sizeof(double));

      tf = fopen(tfile, "r");
      fgets(line, 1024, tf);
      while(fgets(line, 1024, tf) != NULL)
        {
          mult = 1;
          who = 0;
          ptr = 0;
          for(i = 0; i < nmodes; i++ && str != NULL)
            {
              str = strtok (&line[ptr], "\t");
              ptr += strlen(str) + 1;
              indx =  atoi(str)-1;
              singlennz[i] = indx;

              meshind = (gpart[i][indx] / chunksize[i]) % meshdims[i];
              who += meshind*mult;
              mult *= meshdims[i];
            }
          if(who > t->npes)
            printf("yanlis hesapladin %d\n", who);
          memcpy(&allinds[disp[who+1]*nmodes], singlennz, nmodes*sizeof(int));
          str = strtok (&line[ptr], "\t");
			
          allvals[disp[who+1]++] = atof(str);

        }
      fclose(tf);
      free(singlennz);
    }


  MPI_Scatterv(allvals, nnzs, disp, MPI_DOUBLE, t->vals, mynnz, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  if(t->mype == 0)
    {
      for(i = 0; i < t->npes; i++)
        {
          disp[i] *= nmodes;
          nnzs[i] *= nmodes;
        }
      disp[t->npes] *= nmodes;
      disp[t->npes+1] *= nmodes;
    }

  MPI_Scatterv(allinds, nnzs, disp, MPI_INT, t->inds, mynnz*nmodes, MPI_INT, 0, MPI_COMM_WORLD);

  if(t->mype == 0)
    {	
      free(nnzs);
      free(allinds);
      free(allvals);
    }


}


void init_comm (struct tensor *t)
{

  t->comm = (struct comm *)myMalloc(sizeof(struct comm));
  struct comm *co = t->comm;

  co->nrecvwho = (int *)myMalloc(sizeof(int)*t->nmodes);
  co->recvwho = (int **)myMalloc(sizeof(int *)*t->nmodes);
  co->xrecvind = (int **)myMalloc(sizeof(int *)*t->nmodes);
  co->recvind = (int **)myMalloc(sizeof(int *)*t->nmodes);
	
  co->nsendwho = (int *)myMalloc(sizeof(int)*t->nmodes);
  co->sendwho = (int **)myMalloc(sizeof(int *)*t->nmodes);
  co->xsendind = (int **)myMalloc(sizeof(int *)*t->nmodes);
  co->sendind = (int **)myMalloc(sizeof(int *)*t->nmodes);
	
  co->buffer = (double *)myMalloc(sizeof(double)*t->nmodes);

  setintzero(co->nrecvwho, t->nmodes);
  setintzero(co->nsendwho, t->nmodes);

}

void init_stat(struct stat *st, int nmodes)
{
  st->recvvol = (int *)myMalloc(nmodes*sizeof(int));
  st->sendvol = (int *)myMalloc(nmodes*sizeof(int));

  st->recvmsg = (int *)myMalloc(nmodes*sizeof(int));
  st->sendmsg = (int *)myMalloc(nmodes*sizeof(int));

  st->row = (int *)myMalloc(nmodes*sizeof(int));
	
  setintzero(st->recvvol, nmodes);
  setintzero(st->sendvol, nmodes);
  setintzero(st->recvmsg, nmodes);
  setintzero(st->sendmsg, nmodes);
  setintzero(st->row, nmodes);

}


void free_stat(struct stat *st)
{
  if(st->recvvol != NULL)
    free(st->recvvol);

  if(st->sendvol != NULL)
    free(st->sendvol);

  if(st->recvmsg != NULL)
    free(st->recvmsg);

  if(st->sendmsg != NULL)
    free(st->sendmsg);

  if(st->row != NULL)
    free(st->row);

  free(st);

}

void setup_cyclic_communication(struct tensor *t, struct stat *st)
{
  int i, j, k, nmodes, mype, nchunks, meshsize, layersize, *lmap, meshdim, meshind, chunksize, gdim, *gpart, count, *mark, ind, *inds, ptr, lnnz, gp, lp, *map, myrows, *cnts, c, maxbufsize;
  struct comm *co; 

  nmodes = t->nmodes;
  meshsize = t->meshsize;
  mype = t->mype;
	
  init_comm(t);
  co = t->comm;

  t->indmap = (int **)myMalloc(nmodes*sizeof(int *));
  t->ldims = (int *)myMalloc(nmodes*sizeof(int));

  maxbufsize = -1;
  for(i = 0; i < nmodes; i++)
    {

      meshdim = t->meshdims[i];
      meshind = t->meshinds[i];
      chunksize = t->chunksize[i];
      gdim = t->gdims[i];
      gpart = t->gpart[i];
      layersize = t->layersize[i];

      // relabel layer processors
      count = 0;
      lmap = (int *)myMalloc(meshsize*sizeof(int));
      for(j = 0; j < meshsize; j++)
        if((j / chunksize) % meshdim == meshind)
          lmap[j] = count++;
        else
          lmap[j] = -1;

		

      // then indicate the number of rows to be received for mode i 
      mark = (int *)myMalloc(sizeof(int)*gdim);
      setintzero(mark, gdim);

      co->xrecvind[i] = (int *)myMalloc(sizeof(int)*(layersize+2));
      setintzero(co->xrecvind[i], layersize+2);

      inds = t->inds;
      lnnz = t->nnz;
      ptr = i;
      for(k = 0; k < lnnz; k++)
        {
          ind = inds[ptr];
          gp = gpart[ind];
          if(gp != mype)
            if(lmap[gp] != -1)
              {
                lp = lmap[gp];
                if(mark[ind] == 0)
                  {	
                    co->xrecvind[i][lp+2]++;
                    mark[ind] = 1;
                  }
              }
            else
              printf("ERROR\n");
          ptr += nmodes;
        }

      // count local rows
      cnts = (int *)myMalloc((layersize+2)*sizeof(int));
      setintzero(cnts, layersize+2);
      for(j = 0; j < gdim; j++)
        {	
          gp = gpart[j];
          lp = lmap[gp];
          if(lp > layersize)
            printf("DDDD\n");

          if(lp != -1)
            {
              if(gp != mype)
                {
                  if(mark[j] == 1)
                    cnts[lp+2]++;
                }
              else
                cnts[lp+2]++;
            }
        }
      myrows = cnts[lmap[mype]+2];

      //prefix sum
      for(j = 2; j <= layersize+1; j++)
        cnts[j] += cnts[j-1];

      st->row[i] = cnts[layersize+1];
		
      // relabel
      t->indmap[i] = (int *)myMalloc(gdim*sizeof(int));
      map = t->indmap[i];
      for(j = 0; j < gdim; j++)
        {
          gp = gpart[j];
          lp = lmap[gp];
          if(lp != -1)
            {
              if(gp != mype)
                {
                  if(mark[j] == 1)
                    map[j] = cnts[lp+1]++;
                }
              else
                map[j] = cnts[lp+1]++;
            }
          mark[j] = 0;

        }
		
      t->ldims[i] = cnts[layersize+1];

      // go back to receive stuff
      for(j = 2; j <= layersize+1; j++)
        co->xrecvind[i][j] += co->xrecvind[i][j-1];

      co->recvind[i] = (int *)myMalloc(sizeof(int)*co->xrecvind[i][layersize+1]);

      inds = t->inds;
      lnnz = t->nnz;
      ptr = i;

      for(k = 0; k < lnnz; k++)
        {
          ind = inds[ptr];
          gp = gpart[ind];
          if(gp != mype)
            if(lmap[gp] != -1)
              {
                lp = lmap[gp];
                if(mark[ind] == 0)
                  {	
                    co->recvind[i][co->xrecvind[i][lp+1]++] = ind;
                    mark[ind] = 1;
                  }
              }
            else
              printf("ERROR\n");
          ptr += nmodes;

        }
		
      free(mark);

      co->xsendind[i] = (int *)myMalloc(sizeof(int)*(layersize+2));
      setintzero(co->xsendind[i], layersize+2);

      MPI_Request req[layersize];
      for(j = 0; j < layersize; j++)
        if(j != t->layermype[i])
          MPI_Irecv(&(co->xsendind[i][j+1]), 1, MPI_INT, j, 1, t->layercomm[i], j < t->layermype [i] ? &req[j] : &req[j-1]);

      for(j = 0; j < layersize; j++)
        {
          if(j != t->layermype[i])
            {	
              int nrecv = co->xrecvind[i][j+1]-co->xrecvind[i][j]; 
              MPI_Send(&nrecv, 1, MPI_INT, j, 1, t->layercomm[i]);
            }
        }
		
      MPI_Status sta[layersize];
      MPI_Waitall(layersize-1, req, sta);


      for(j = 1; j < layersize+1; j++)
        co->xsendind[i][j] += co->xsendind[i][j-1];

      co->sendind[i] = (int *)myMalloc(sizeof(int)*co->xsendind[i][layersize]); 

      co->recvwho[i] = (int *)myMalloc(sizeof(int)*layersize);
      setintzero(co->recvwho[i], layersize);

      for(j = 0; j < layersize; j++)
        if(co->xrecvind[i][j+1]-co->xrecvind[i][j] > 0)
          co->recvwho[i][co->nrecvwho[i]++] = j;

      co->sendwho[i] = (int *)myMalloc(sizeof(int)*layersize);
      setintzero(co->sendwho[i], layersize);

      for(j = 0; j < layersize; j++)
        if(co->xsendind[i][j+1]-co->xsendind[i][j] > 0)
          co->sendwho[i][co->nsendwho[i]++] = j;		

      for(j = 0; j < co->nsendwho[i]; j++)
        {
          int p = co->sendwho[i][j];
          MPI_Irecv(&(co->sendind[i][co->xsendind[i][p]]), co->xsendind[i][p+1]-co->xsendind[i][p], MPI_INT, p, 2, t->layercomm[i], &req[j]);
        }

      for(j = 0; j < co->nrecvwho[i]; j++)
        {	
          int p = co->recvwho[i][j];
          MPI_Send(&(co->recvind[i][co->xrecvind[i][p]]), co->xrecvind[i][p+1]-co->xrecvind[i][p], MPI_INT, p, 2, t->layercomm[i]);

        }
      MPI_Waitall(co->nsendwho[i], req, sta);

      st->recvvol[i] = co->xrecvind[i][layersize];
      st->sendvol[i] = co->xsendind[i][layersize];
      st->recvmsg[i] = co->nrecvwho[i];
      st->sendmsg[i] = co->nsendwho[i];

      // we dont need recvind indices anymore, do we?
      for(j = lmap[mype]; j < layersize; j++)
        co->xrecvind[i][j+1] += myrows;

      // localize indices in tensor
      lnnz = t->nnz;
      inds = t->inds;
			
      map = t->indmap[i];
      ptr = i;
      for(j = 0; j < lnnz; j++)
        {	
          inds[ptr] = map[inds[ptr]];
          ptr += nmodes;
        }
		
      // localize indices in sendind
      for(j = 0; j < co->xsendind[i][layersize]; j++)
        co->sendind[i][j] = map[co->sendind[i][j]];


      free(lmap);
      free(cnts);

      if(co->xrecvind[i][layersize] > maxbufsize)
        maxbufsize = co->xrecvind[i][layersize];
      if(co->xsendind[i][layersize] > maxbufsize)
        maxbufsize = co->xsendind[i][layersize];

    }

  st->nnz = t->nnz;

  co->buffer = (double *)myMalloc(sizeof(double)*maxbufsize*t->cprank);

}

void setup_ckbd_communication(struct tensor *t, struct stat *st)
{
  int i, j, k, nmodes, layermype, layersize, meshind, gdim, *intrapart, *interpart, count, *mark, ind, *inds, ptr, lnnz, p, *map, myrows, *cnts,  maxbufsize;
  struct comm *co; 

  nmodes = t->nmodes;

  init_comm(t);
  co = t->comm;

  t->indmap = (int **)myMalloc(nmodes*sizeof(int *));
  t->ldims = (int *)myMalloc(nmodes*sizeof(int));

  maxbufsize = -1;
  for(i = 0; i < nmodes; i++)
    {
      layermype = t->layermype[i];
      meshind = t->meshinds[i];
      gdim = t->gdims[i];
      intrapart = t->intrapart[i];
      interpart = t->interpart[i];
      layersize = t->layersize[i];

      // then indicate the number of rows to be received for mode i 
      mark = (int *)myMalloc(sizeof(int)*gdim);
      setintzero(mark, gdim);

      co->xrecvind[i] = (int *)myMalloc(sizeof(int)*(layersize+2));
      setintzero(co->xrecvind[i], layersize+2);

      inds = t->inds;
      lnnz = t->nnz;
      ptr = i;
      for(k = 0; k < lnnz; k++)
        {
          ind = inds[ptr];
          p = intrapart[ind];
          if(p != layermype)
            if(mark[ind] == 0)
              {	
                co->xrecvind[i][p+2]++;
                mark[ind] = 1;
              }
          ptr += nmodes;
        }

      // count local rows
      cnts = (int *)myMalloc((layersize+2)*sizeof(int));
      setintzero(cnts, layersize+2);

      for(j = 0; j < gdim; j++)
        {	
          if(interpart[j] == meshind)
            {
              p = intrapart[j];
              if(p > layersize)
                printf("DDDD\n");
              
              if(p != layermype)
                {
                  if(mark[j] == 1)
                    cnts[p+2]++;
                }
              else
                cnts[p+2]++;
            }

        }
      myrows = cnts[layermype+2];

      //prefix sum
      for(j = 2; j <= layersize+1; j++)
        cnts[j] += cnts[j-1];

      st->row[i] = myrows;

      // relabel
      t->indmap[i] = (int *)myMalloc(gdim*sizeof(int));
      map = t->indmap[i];
      for(j = 0; j < gdim; j++)
        {
          if(interpart[j] == meshind)
            {
              p = intrapart[j];
              if(p != layermype)
                {
                  if(mark[j] == 1)
                    map[j] = cnts[p+1]++;
                }
              else
                map[j] = cnts[p+1]++;
            }
          mark[j] = 0;

        }
      t->ldims[i] = cnts[layersize+1];

      // go back to receive stuff
      for(j = 2; j <= layersize+1; j++)
        co->xrecvind[i][j] += co->xrecvind[i][j-1];

      co->recvind[i] = (int *)myMalloc(sizeof(int)*co->xrecvind[i][layersize+1]);

      ptr = i;
      for(k = 0; k < lnnz; k++)
        {
          ind = inds[ptr];
          p = intrapart[ind];
          if(p != layermype)
            if(mark[ind] == 0)
              {	
                co->recvind[i][co->xrecvind[i][p+1]++] = ind;
                mark[ind] = 1;
              }
          ptr += nmodes;
        }
      free(mark);

      co->xsendind[i] = (int *)myMalloc(sizeof(int)*(layersize+2));
      setintzero(co->xsendind[i], layersize+2);

      MPI_Request req[layersize];
      for(j = 0; j < layersize; j++)
        if(j != t->layermype[i])
          MPI_Irecv(&(co->xsendind[i][j+1]), 1, MPI_INT, j, 1, t->layercomm[i], j < t->layermype [i] ? &req[j] : &req[j-1]);

      for(j = 0; j < layersize; j++)
        {
          if(j != t->layermype[i])
            {	
              int nrecv = co->xrecvind[i][j+1]-co->xrecvind[i][j]; 
              MPI_Send(&nrecv, 1, MPI_INT, j, 1, t->layercomm[i]);
            }
        }

      MPI_Status sta[layersize];
      MPI_Waitall(layersize-1, req, sta);

      for(j = 1; j < layersize+1; j++)
        co->xsendind[i][j] += co->xsendind[i][j-1];

      co->sendind[i] = (int *)myMalloc(sizeof(int)*co->xsendind[i][layersize]); 

      co->recvwho[i] = (int *)myMalloc(sizeof(int)*layersize);
      setintzero(co->recvwho[i], layersize);

      for(j = 0; j < layersize; j++)
        if(co->xrecvind[i][j+1]-co->xrecvind[i][j] > 0)
          co->recvwho[i][co->nrecvwho[i]++] = j;

      co->sendwho[i] = (int *)myMalloc(sizeof(int)*layersize);
      setintzero(co->sendwho[i], layersize);

      for(j = 0; j < layersize; j++)
        if(co->xsendind[i][j+1]-co->xsendind[i][j] > 0)
          co->sendwho[i][co->nsendwho[i]++] = j;		

      for(j = 0; j < co->nsendwho[i]; j++)
        {
          int p = co->sendwho[i][j];
          MPI_Irecv(&(co->sendind[i][co->xsendind[i][p]]), co->xsendind[i][p+1]-co->xsendind[i][p], MPI_INT, p, 2, t->layercomm[i], &req[j]);
        }

      for(j = 0; j < co->nrecvwho[i]; j++)
        {	
          int p = co->recvwho[i][j];
          MPI_Send(&(co->recvind[i][co->xrecvind[i][p]]), co->xrecvind[i][p+1]-co->xrecvind[i][p], MPI_INT, p, 2, t->layercomm[i]);

        }
      MPI_Waitall(co->nsendwho[i], req, sta);

      st->recvvol[i] = co->xrecvind[i][layersize];
      st->sendvol[i] = co->xsendind[i][layersize];
      st->recvmsg[i] = co->nrecvwho[i];
      st->sendmsg[i] = co->nsendwho[i];

      // we dont need recvind indices anymore, do we?
      for(j = layermype; j < layersize; j++)
        co->xrecvind[i][j+1] += myrows;

      // localize indices in tensor
      ptr = i;
      for(j = 0; j < lnnz; j++)
        {	
          inds[ptr] = map[inds[ptr]];
          ptr += nmodes;
        }
		
      // localize indices in sendind
      for(j = 0; j < co->xsendind[i][layersize]; j++)
        co->sendind[i][j] = map[co->sendind[i][j]];


      free(cnts);

      if(co->xrecvind[i][layersize] > maxbufsize)
        maxbufsize = co->xrecvind[i][layersize];
      if(co->xsendind[i][layersize] > maxbufsize)
        maxbufsize = co->xsendind[i][layersize];

    }

  st->nnz = t->nnz;
  co->buffer = (double *)myMalloc(sizeof(double)*maxbufsize*t->cprank);

}


void setup_fg_communication(struct tensor *t, struct stat *st)
{
  int i, j, k, nmodes, gdim, *interpart, count, *mark, ind, *inds, ptr, lnnz, p, *map, myrows, *cnts,  maxbufsize, mype, size;
  struct comm *co; 

  nmodes = t->nmodes;
  mype = t->mype;
  size = t->npes;

  init_comm(t);
  co = t->comm;

  t->indmap = (int **)myMalloc(nmodes*sizeof(int *));
  t->ldims = (int *)myMalloc(nmodes*sizeof(int));

  maxbufsize = -1;
  for(i = 0; i < nmodes; i++)
    {
      gdim = t->gdims[i];
      interpart = t->interpart[i];

      // then indicate the number of rows to be received for mode i 
      mark = (int *)myMalloc(sizeof(int)*gdim);
      setintzero(mark, gdim);

      co->xrecvind[i] = (int *)myMalloc(sizeof(int)*(size+2));
      setintzero(co->xrecvind[i], size+2);

      inds = t->inds;
      lnnz = t->nnz;
      ptr = i;
      for(k = 0; k < lnnz; k++)
        {
          ind = inds[ptr];
          p = interpart[ind];
          if(p != mype)
            if(mark[ind] == 0)
              {	
                co->xrecvind[i][p+2]++;
                mark[ind] = 1;
              }
          ptr += nmodes;
        }

      // count local rows
      cnts = (int *)myMalloc((size+2)*sizeof(int));
      setintzero(cnts, size+2);

      for(j = 0; j < gdim; j++)
        {
          p = interpart[j];
          if(p != mype)
            {
              if(mark[j] == 1)
                cnts[p+2]++;
            }
          else
            cnts[p+2]++;
        }

      myrows = cnts[mype+2];

      //prefix sum
      for(j = 2; j <= size+1; j++)
        cnts[j] += cnts[j-1];
      
      st->row[i] = myrows;

      // relabel
      t->indmap[i] = (int *)myMalloc(gdim*sizeof(int));
      map = t->indmap[i];
      for(j = 0; j < gdim; j++)
        {
          p = interpart[j];
          if(p != mype)
            {
              if(mark[j] == 1)
                map[j] = cnts[p+1]++;
            }
          else
            map[j] = cnts[p+1]++;
          
          mark[j] = 0;
          
        }
      t->ldims[i] = cnts[size+1];
      
      // go back to receive stuff
      for(j = 2; j <= size+1; j++)
        co->xrecvind[i][j] += co->xrecvind[i][j-1];

      co->recvind[i] = (int *)myMalloc(sizeof(int)*co->xrecvind[i][size+1]);

      ptr = i;
      for(k = 0; k < lnnz; k++)
        {
          ind = inds[ptr];
          p = interpart[ind];
          if(p != mype)
            if(mark[ind] == 0)
              {	
                co->recvind[i][co->xrecvind[i][p+1]++] = ind;
                mark[ind] = 1;
              }
          ptr += nmodes;
        }
      free(mark);

      co->xsendind[i] = (int *)myMalloc(sizeof(int)*(size+2));
      setintzero(co->xsendind[i], size+2);

      MPI_Request req[size];
      for(j = 0; j < size; j++)
        if(j != mype)
          MPI_Irecv(&(co->xsendind[i][j+1]), 1, MPI_INT, j, 1, MPI_COMM_WORLD, j < mype ? &req[j] : &req[j-1]);

      for(j = 0; j < size; j++)
        {
          if(j != mype)
            {	
              int nrecv = co->xrecvind[i][j+1]-co->xrecvind[i][j]; 
              MPI_Send(&nrecv, 1, MPI_INT, j, 1, MPI_COMM_WORLD);
            }
        }
      
      MPI_Status sta[size];
      MPI_Waitall(size-1, req, sta);

      for(j = 1; j < size+1; j++)
        co->xsendind[i][j] += co->xsendind[i][j-1];

      co->sendind[i] = (int *)myMalloc(sizeof(int)*co->xsendind[i][size]); 

      co->recvwho[i] = (int *)myMalloc(sizeof(int)*size);
      setintzero(co->recvwho[i], size);

      for(j = 0; j < size; j++)
        if(co->xrecvind[i][j+1]-co->xrecvind[i][j] > 0)
          co->recvwho[i][co->nrecvwho[i]++] = j;

      co->sendwho[i] = (int *)myMalloc(sizeof(int)*size);
      setintzero(co->sendwho[i], size);

      for(j = 0; j < size; j++)
        if(co->xsendind[i][j+1]-co->xsendind[i][j] > 0)
          co->sendwho[i][co->nsendwho[i]++] = j;		

      for(j = 0; j < co->nsendwho[i]; j++)
        {
          int p = co->sendwho[i][j];
          MPI_Irecv(&(co->sendind[i][co->xsendind[i][p]]), co->xsendind[i][p+1]-co->xsendind[i][p], MPI_INT, p, 2, MPI_COMM_WORLD, &req[j]);
        }

      for(j = 0; j < co->nrecvwho[i]; j++)
        {	
          int p = co->recvwho[i][j];
          MPI_Send(&(co->recvind[i][co->xrecvind[i][p]]), co->xrecvind[i][p+1]-co->xrecvind[i][p], MPI_INT, p, 2, MPI_COMM_WORLD);
        }
      MPI_Waitall(co->nsendwho[i], req, sta);

      st->recvvol[i] = co->xrecvind[i][size];
      st->sendvol[i] = co->xsendind[i][size];
      st->recvmsg[i] = co->nrecvwho[i];
      st->sendmsg[i] = co->nsendwho[i];

      // we dont need recvind indices anymore, do we?
      for(j = mype; j < size; j++)
        co->xrecvind[i][j+1] += myrows;
      
      // localize indices in tensor
      ptr = i;
      for(j = 0; j < lnnz; j++)
        {	
          inds[ptr] = map[inds[ptr]];
          ptr += nmodes;
        }
		
      // localize indices in sendind
      for(j = 0; j < co->xsendind[i][size]; j++)
        co->sendind[i][j] = map[co->sendind[i][j]];
      
      
      free(cnts);

      if(co->xrecvind[i][size] > maxbufsize)
        maxbufsize = co->xrecvind[i][size];
      if(co->xsendind[i][size] > maxbufsize)
        maxbufsize = co->xsendind[i][size];

    }

  st->nnz = t->nnz;
  co->buffer = (double *)myMalloc(sizeof(double)*maxbufsize*t->cprank);

}


//assumes the first line of the tensor starts with # and denotes dimension info
int read_dimensions(char* tfile, struct tensor *t){

  FILE *tf = fopen(tfile, "r");
  char line[1024];
  fgets(line, 1024, tf);
  fclose(tf);

  int nmodes = -1, ptr = 1, i;
  char * str;

  str = strtok (line, "\t");
  while (str != NULL){
    nmodes++; 
    str = strtok (NULL, "\t");
  }

  t->gdims = (int *)myMalloc(sizeof(int)*nmodes);

  for(i = 0; i < nmodes; i++ && str != NULL)
    {
      str = strtok (&line[ptr], "\t");
      t->gdims[i] = atoi(str);
      ptr += strlen(str) + 1;
    }

  str = strtok (&line[ptr], "\t");
  t->gnnz = atoi(str);
	
  t->nmodes = nmodes;

  return 0;
}


int convert(int value)
{
  char arr[4];
  arr[0] = (char)(value >> 24);
  arr[1] = (char)(value >> 16);
  arr[2] = (char)(value >> 8);
  arr[3] = (char)(value);

  return arr[3] << 24 | (arr[2] & 0xFF) << 16 | (arr[1] & 0xFF) << 8 | (arr[0] & 0xFF);

}

int read_dimensions_bin_endian(char* tfile, struct tensor *t){

  int nmodes, j;
  FILE *tf = fopen(tfile, "rb");

  fread(&nmodes, sizeof(int), 1, tf);
  nmodes = convert(nmodes);

  t->gdims = (int *)myMalloc(sizeof(int)*nmodes);
  fread(t->gdims, sizeof(int), nmodes, tf);
  for(j = 0; j < nmodes+2; j++)
    t->gdims[j] = convert(t->gdims[j]);

  fread(&(t->gnnz), sizeof(int), 1, tf);
  t->gnnz = convert(t->gnnz);
  t->nmodes = nmodes;

  fclose(tf);

  return 0;
}


//assumes nmodes - dim1, dim2, .. dimnmodes - totalnnz - mynnz
int read_dimensions_bin(char* tfile, struct tensor *t){

  int nmodes;
  FILE *tf = fopen(tfile, "rb");

  fread(&nmodes, sizeof(int), 1, tf);
  t->gdims = (int *)myMalloc(sizeof(int)*nmodes);
  fread(t->gdims, sizeof(int), nmodes, tf);
  fread(&(t->gnnz), sizeof(int), 1, tf);
  t->nmodes = nmodes;

  fclose(tf);

  return 0;
}



void read_cyclic_partition(char *partfile, struct tensor *t)
{
  int i, j;
  int nmodes = t->nmodes;
  int *gdims = t->gdims;

  FILE *fpart = fopen(partfile, "r");
  t->gpart = (int **)myMalloc(sizeof(int*)*nmodes);
  for(i = 0; i < nmodes; i++)
    {
      t->gpart[i] = (int *)myMalloc(sizeof(int)*gdims[i]);
      for(j = 0; j < gdims[i]; j++)
        fscanf(fpart, "%d\n", &(t->gpart[i][j]));
    }
  fclose(fpart);
}

void compute_mesh_dim(struct tensor *t)
{
  int i, j, nfac, *fac, number, nmodes, target, furthest, *dist;

  number = t->npes;
  nfac= (int)log2((double)number)+1;
  fac = (int *)myMalloc(sizeof(int)*nfac);

  nfac = 0;
  for(i = 2; i <= number; i++)
    while(number % i == 0 )
      {
        fac[nfac++] = i;
        number = number / i;
      }

  nmodes = t->nmodes;
  target = 0;

  t->meshdims = (int *)myMalloc(sizeof(int)*nmodes);
  for(i = 0; i < nmodes; i++)
    {
      t->meshdims[i] = 1;
      target += t->gdims[i];
    }

  target = target/t->npes;

  for(i = nfac-1; i >= 0; i--)
    {
      furthest = 0;
      dist = (int *)myMalloc(nmodes*sizeof(int));
      setintzero(dist, nmodes);
		
      for(j = 0; j < nmodes; j++)
        {
          dist[j] = t->gdims[j]/t->meshdims[j] - target;
          if(dist[j] > dist[furthest])
            furthest = j;
        }

      t->meshdims[furthest] = t->meshdims[furthest]*fac[i];
      free(dist);
    }
  t->meshsize = t->npes;

  free(fac);

}

void parse_mesh_str(struct tensor *t, char meshstr[1024])
{
  int nmodes = 0, ptr = 0, i;
  char * str;

  str = strtok (meshstr, "x");
  while (str != NULL){
    nmodes++; 
    str = strtok (NULL, "x");
  }

  if(nmodes != t->nmodes)
    {
      if(t->mype == 0)
        printf("Number of modes of the tensor in the file does not match the number of modes of the parsed mesh dimensionality string\n");

      free_tensor(t);
      MPI_Finalize();
      exit(1);
    }

  t->meshdims = (int *)myMalloc(sizeof(int)*nmodes);
  t->meshsize = 1;

  for(i = 0; i < nmodes; i++ && str != NULL)
    {
      str = strtok (&meshstr[ptr], "x");
      t->meshdims[i] = atoi(str);
      t->meshsize *= t->meshdims[i];
      ptr += strlen(str) + 1;
    }


}

void split_communicators(struct tensor *t)
{
  int i, lid, mult, nmodes, *meshdims, *meshinds;
	
  nmodes = t->nmodes;

  t->meshinds = (int *)myMalloc(sizeof(int)*nmodes);
  t->layercomm = (MPI_Comm *)myMalloc(sizeof(MPI_Comm)*nmodes);
  t->layermype = (int *)myMalloc(sizeof(int)*nmodes);
  t->layersize = (int *)myMalloc(sizeof(int)*nmodes);

  mult = t->mype;
  meshdims = t->meshdims;
  meshinds = t->meshinds;

  // get mesh indices of me
  for(i = 0; i < nmodes; i++)
    {	
      meshinds[i] = mult % meshdims[i];
      mult = mult / meshdims[i];
    }

  for(i = 0; i < nmodes; i++)
    {
      lid = meshinds[i];

      MPI_Comm_split(MPI_COMM_WORLD, lid, 0, &(t->layercomm[i]));
      MPI_Comm_rank(t->layercomm[i], &(t->layermype[i]));
      MPI_Comm_size(t->layercomm[i], &(t->layersize[i]));
    }

}

int init_cyclic_tensor(char tensorfile[], char meshstr[], struct tensor *t)
{
  int c;
  char tname[1024];
	
  read_dimensions(tensorfile, t);

  if(strcmp(meshstr, "auto") == 0)
    compute_mesh_dim(t);
  else
    parse_mesh_str(t, meshstr);

  split_communicators(t);

  return 0;
}

int read_cyclic_tensor(char *tensorfile, char *partfile, char meshstr[], struct tensor *t)
{

  init_cyclic_tensor(tensorfile, meshstr, t);
	
  read_cyclic_partition(partfile, t);

  get_chunk_info(t);

  read_cyclic_tensor_nonzeros(tensorfile, t);
	
  return 0;
}       



void printusage(char *exec)
{
  printf("usage: %s [options] tensorfile\n", exec);
  printf("\t-p partitionfile: (char *) in the partition file, indices numbered in the mode order\n");
  printf("\t-m meshstring: (char *) number of processors in each dimension. e.g.: 4x2x2\n");
  printf("\t-r rank: (int) rank of CP decomposition. default: 16\n");
  exit(1);
}





void substring(char *text, char out[1024])
{
  char *ptr = text;
  char *prevptr = NULL;

  while( (ptr = strstr(ptr,"/")))
    {
      prevptr = ptr++;
    }
  prevptr++;


  int sl = strlen(prevptr);
  strncpy(out, prevptr, sl);
  out[sl] = '\0';

}

void init_param(int argc, char *argv[], char tensorfile[], char partfile[], char meshstr[], struct tensor *t, int *niters, int *endian)
{
  //set default values
  t->ckbd = 1;
  t->cprank = 16;
  t->fiber = 1;
  t->alltoall = 0;
  //*niters = 10;
  *endian = 0;
  strcpy(meshstr,"auto");
  strcpy(partfile, "");

  int c;
  //while ((c = getopt(argc, argv, "c:p:m:r:f:a:i:e:")) != -1)
  while ((c = getopt(argc, argv, "c:p:m:r:f:a:e:")) != -1)
    {
      switch (c)
        {
        case 'c':       t->ckbd = atoi(optarg); 
          break;
        case 'p': 	strcpy(partfile, optarg);
          break;
        case 'm':	strcpy(meshstr, optarg);
          break;
        case 'r':       t->cprank = atoi(optarg); 
          break;
        case 'f':       t->fiber = atoi(optarg); 
          break; 
        case 'a':       t->alltoall = atoi(optarg); 
          break;
        //case 'i':       *niters = atoi(optarg); 
//          break;
        case 'e':       *endian = atoi(optarg); 
          break;
        }
    }

  if(argc <= optind)
    printusage(argv[0]);
  if(strcmp(partfile, "") == 0)
    printf("A partition file must be provided\n");

  sprintf(tensorfile, "%s", argv[optind]);


}




void free_comm(struct comm *c, int nmodes)
{
  int i;

  if(c->nrecvwho != NULL)
    free(c->nrecvwho);

  if(c->recvwho != NULL)
    {
      for(i = 0; i < nmodes; i++)
        c->recvwho[i];
      free(c->recvwho);
    }

  if(c->xrecvind != NULL)
    {
      for(i = 0; i < nmodes; i++)
        c->xrecvind[i];
      free(c->xrecvind);
    }

  if(c->recvind != NULL)
    {
      for(i = 0; i < nmodes; i++)
        c->recvind[i];
      free(c->recvind);
    }

  if(c->nsendwho != NULL)
    free(c->nsendwho);

  if(c->sendwho != NULL)
    {
      for(i = 0; i < nmodes; i++)
        c->sendwho[i];
      free(c->sendwho);
    }

  if(c->xsendind != NULL)
    {
      for(i = 0; i < nmodes; i++)
        c->xsendind[i];
      free(c->xsendind);
    }

  if(c->sendind != NULL)
    {
      for(i = 0; i < nmodes; i++)
        c->sendind[i];
      free(c->sendind);
    }

  if(c->buffer != NULL)
    free(c->buffer);

  free(c);


}

void free_tensor(struct tensor *t)
{
  int i;

  if(t->gdims != NULL)
    free(t->gdims);

  if(t->ldims != NULL)
    free(t->ldims);

  if(t->indmap != NULL)
    {
      for(i = 0; i < t->nmodes; i++)
        free(t->indmap[i]);
    }

  if(t->gpart != NULL)
    {	
      for(i = 0; i < t->nmodes; i++)
        free(t->gpart[i]);
      free(t->gpart);
    }
  if(t->interpart != NULL)
    {	
      for(i = 0; i < t->nmodes; i++)
        free(t->interpart[i]);
      free(t->interpart);
    }
  if(t->intrapart != NULL)
    {	
      for(i = 0; i < t->nmodes; i++)
        free(t->intrapart[i]);
      free(t->intrapart);
    }

  if(t->meshdims != NULL)
    free(t->meshdims);

  if(t->meshinds != NULL)
    free(t->meshinds);

  if(t->chunksize != NULL)
    free(t->chunksize);

  if(t->inds != NULL)
    free(t->inds);

  if(t->vals != NULL)
    free(t->vals);

  if(t->layercomm != NULL)
    free(t->layercomm);
	
  if(t->layermype != NULL)
    free(t->layermype);

  if(t->layersize != NULL)
    free(t->layersize);

  if(t->comm != NULL)
    free_comm(t->comm, t->nmodes);

  if(t->mat != NULL)
    {
      for(i = 0; i < t->nmodes; i++)
        free(t->mat[i]);
      free(t->mat);
    }

  if(t->uTu != NULL)
    free(t->uTu);

  free(t);
}

void free_fibertensor(struct fibertensor *ft, int nmodes)
{
  int i, j;
  if(ft->fibermode != NULL)
    free(ft->fibermode);

  if(ft->order != NULL)
    {
      for(i = 0; i < nmodes; i++)
        free(ft->order[i]);
      free(ft->order);
    }

  if(ft->topmostcnt != NULL)
    free(ft->topmostcnt);

  if(ft->xfibers != NULL)
    {
      for(i = 0; i < nmodes; i++)
        {
          for(j = 0; j < nmodes-1; j++)
            free(ft->xfibers[i][j]);
          free(ft->xfibers[i]);
        }
      free(ft->xfibers);
    }

  if(ft->lfibers != NULL)
    free(ft->lfibers);

  if(ft->slfibers != NULL)
    free(ft->slfibers);

  if(ft->lvals != NULL)
    free(ft->lvals);

  if(ft->slvals != NULL)
    free(ft->slvals);
}


void init_tensor(struct tensor *t)
{
  t->gdims = NULL;
  t->ldims = NULL;
  t->indmap = NULL;
  t->gpart = NULL;
  t->interpart = NULL;
  t->intrapart = NULL;
  t->meshdims = NULL;
  t->meshinds = NULL;
  t->chunksize = NULL;
  t->inds = NULL;
  t->vals = NULL;
  t->comm = NULL;
  t->layercomm = NULL;
  t->layermype = NULL;
  t->layersize = NULL;
  t->mat = NULL;
  t->uTu = NULL;

}

void init_fibertensor(struct fibertensor *ft)
{
  ft->fibermode = NULL;
  ft->order = NULL;
  ft->topmostcnt = NULL;
  ft->xfibers = NULL;
  ft->lfibers = NULL;
  ft->slfibers = NULL;
  ft->lvals = NULL;
  ft->slvals = NULL;
}

void init_matrices(struct tensor *t)
{
  int i, j, k, l, nmodes, cprank, size, base;
  double *mat, v, *tmp, w;

  //	srand (time(NULL));

  nmodes = t->nmodes;
  cprank = t->cprank;
  tmp = (double *)myMalloc(nmodes*cprank*cprank*sizeof(double));
  t->mat = (double **)myMalloc(nmodes*sizeof(double *));

  for(i = 0; i < nmodes; i++)
    {	
      size = t->ldims[i]*t->cprank;
      //      posix_memalign(&(t->mat[i]), 64, size*sizeof(double));
      //t->mat[i] = (double *)myMalloc(size*sizeof(double));
      t->mat[i] = (double *)myMalloc(size*sizeof(double));
      mat = t->mat[i];
      for(j = 0; j < size; j++)
        {
          /*			mat[j] = 3.0 * (((double)rand()+1) / (double) RAND_MAX);
                        if(rand() % 2 == 0)
                        mat[j] *= -1;
          */
          mat[j] = 1.0;
				
        }
      base = cprank*cprank*i;
      size = t->ldims[i];
      for(j = 0; j < cprank; j++)
        {
          v = 0;
          for(l = 0; l < size; l++)
            {	
              w = mat[l*cprank+j];
              v += w*w;
            }
          tmp[base+j*cprank+j] = v;
			
          for(k = j+1; k < cprank; k++)
            {
              v = 0;
              for(l = 0; l < size; l++)
                v += mat[l*cprank+j]*mat[l*cprank+k];

              tmp[base+k*cprank+j] = v;
              tmp[base+j*cprank+k] = v;
            }
        }
    }


  t->uTu = (double *)myMalloc(nmodes*cprank*cprank*sizeof(double));
	
  MPI_Allreduce(tmp,t->uTu, cprank*cprank*nmodes, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

  free(tmp);
}

void mttkrp_nnz(struct tensor *t, int mode, double *matm)
{
  int c, i, j, k, size, cprank, nmodes, iwrite, iread, *inds, nnz;
  double *vals, v, *acc, *mat, **mats;

  nmodes = t->nmodes;
  cprank = t->cprank;
  mats = t->mat;

  acc = (double *)myMalloc(sizeof(double)*cprank);

	
  nnz = t->nnz;
  inds = t->inds;
  vals = t->vals;
		
  for(i = 0; i < nnz; i++)
    {
      v = vals[i];
      for(k = 0; k < cprank; k++)
        acc[k] = v;

      for(j = 0; j < nmodes; j++)
        {
          if(j != mode)
            {
              iread = inds[i*nmodes+j]*cprank;
              mat = mats[j];
				
              for(k = 0; k < cprank; k++)
                acc[k] *= mat[iread++];
            }	
        }

      iwrite = inds[i*nmodes+mode]*cprank;
      for(k = 0; k < cprank; k++)
        matm[iwrite++] += acc[k];
    }
	
  free(acc);
}

void mttkrp_fiber_3(struct tensor *t, struct fibertensor *ft, int mode, double *matm)
{
  int j, nmodes, cprank, cnt, *xfibers0, *xfibers1, *fibers, i0, st0, en0, base0, i1, st1, en1, base1, i2, base2;
  double *mat1, *mat2, *vals, *acc;

  nmodes = t->nmodes;
  cprank = t->cprank;

  xfibers0 = ft->xfibers[mode][0];
  xfibers1 = ft->xfibers[mode][1];

  cnt = ft->topmostcnt[mode];
	
  mat1 = t->mat[ft->order[mode][1]];
  mat2 = t->mat[ft->order[mode][2]];

  if(mode == ft->lmode)
    {
      fibers = ft->slfibers;
      vals = ft->slvals;
    }
  else
    {
      fibers = ft->lfibers;
      vals = ft->lvals;
    }

  acc = (double *)myMalloc(sizeof(double)*cprank);


  for(i0 = 0; i0 < cnt; i0++)
    {
      base0 = i0*2;
      st0 = xfibers0[base0];
      en0 = xfibers0[base0+2];
      base0 = xfibers0[base0+1]*cprank;

      for(i1 = st0; i1 < en0; i1++)
        {
          base1 = i1*2;
          st1 = xfibers1[base1];
          en1 = xfibers1[base1+2];
          base1 = xfibers1[base1+1]*cprank;

          memset(acc, 0, sizeof(double)*cprank);

          for(i2 = st1; i2 < en1; i2++)
            {
              base2 = fibers[i2]*cprank; 
              for(j = 0; j < cprank; j++)
                acc[j] += mat2[base2+j]*vals[i2];
            }
			
          for(j = 0; j < cprank; j++)
            matm[base0+j] += acc[j]*mat1[base1+j];
        }

    }

  free(acc);
}

void mttkrp_fiber_4(struct tensor *t, struct fibertensor *ft, int mode, double *matm)
{
  int j, nmodes, cprank, cnt, *xfibers0, *xfibers1, *xfibers2, *fibers, i0, st0, en0, i1, st1, en1, i2, st2, en2, en3, i3, base0, base1, base2, base3;
  double *mat1, *mat2, *mat3, *vals, *acc, *acc2;

  nmodes = t->nmodes;
  cprank = t->cprank;

  xfibers0 = ft->xfibers[mode][0];
  xfibers1 = ft->xfibers[mode][1];
  xfibers2 = ft->xfibers[mode][2];

  cnt = ft->topmostcnt[mode];
	
  mat1 = t->mat[ft->order[mode][1]];
  mat2 = t->mat[ft->order[mode][2]];
  mat3 = t->mat[ft->order[mode][3]];

  if(mode == ft->lmode)
    {
      fibers = ft->slfibers;
      vals = ft->slvals;
    }
  else
    {
      fibers = ft->lfibers;
      vals = ft->lvals;
    }

  acc = (double *)myMalloc(sizeof(double)*cprank);
  acc2 = (double *)myMalloc(sizeof(double)*cprank); 
	

  for(i0 = 0; i0 < cnt; i0++)
    {
      base0 = i0*2;
      st0 = xfibers0[base0];
      en0 = xfibers0[base0+2];
      base0 = xfibers0[base0+1]*cprank;

      for(i1 = st0; i1 < en0; i1++)
        {
          base1 = i1*2;
          st1 = xfibers1[base1];
          en1 = xfibers1[base1+2];
          base1 = xfibers1[base1+1]*cprank;

          memset(acc, 0, sizeof(double)*cprank);
          for(i2 = st1; i2 < en1; i2++)
            {
              base2 = i2*2;
              st2 = xfibers2[base2];
              en2 = xfibers2[base2+2];
              base2 = xfibers2[base2+1]*cprank;

              memset(acc2, 0, sizeof(double)*cprank);
              for(i3 = st2; i3 < en2; i3++)
                {
                  base3 = fibers[i3]*cprank;
                  for(j = 0; j < cprank; j++)
                    acc2[j] += mat3[base3+j]*vals[i3];
                }
				
              for(j = 0; j < cprank; j++)
                acc[j] += acc2[j]*mat2[base2+j];
				

            }
          for(j = 0; j < cprank; j++)
            matm[base0+j] += acc[j]*mat1[base1+j];
			
        }

    }

  free(acc);
  free(acc2);
}


double mttkrp_nnz_stat(struct tensor *t, int mode, double *matm, int niters)
{
  int it, c, i, j, k, size, cprank, nmodes, iwrite, iread, *inds, nnz;
  double *vals, v, *acc, *mat, **mats, time;

  nmodes = t->nmodes;
  cprank = t->cprank;
  mats = t->mat;

  acc = (double *)myMalloc(sizeof(double)*cprank);
  nnz = t->nnz;
  inds = t->inds;
  vals = t->vals;


  MPI_Barrier(MPI_COMM_WORLD);
  time = (double) get_wc_time();
	
  for(it = 0; it < niters; it++)
    {
	
      for(i = 0; i < nnz; i++)
        {
          v = vals[i];
          for(k = 0; k < cprank; k++)
            acc[k] = v;

          for(j = 0; j < nmodes; j++)
            {
              if(j != mode)
                {
                  iread = inds[i*nmodes+j]*cprank;
                  mat = mats[j];
				
                  for(k = 0; k < cprank; k++)
                    acc[k] *= mat[iread++];
                }	
            }

          iwrite = inds[i*nmodes+mode]*cprank;
          for(k = 0; k < cprank; k++)
            matm[iwrite++] += acc[k];
        }
    }	
  MPI_Barrier(MPI_COMM_WORLD);
  time = ((double) get_wc_time() - time)/niters;

  free(acc);

  return time;
}

double mttkrp_fiber_3_stat(struct tensor *t, struct fibertensor *ft, int mode, double *matm, int niters, int *cnt_st)
{
  int it, j, nmodes, cprank, cnt, *xfibers0, *xfibers1, *fibers, i0, st0, en0, base0, i1, st1, en1, base1, i2, base2;
  double *mat1, *mat2, *vals, *acc, time;

  nmodes = t->nmodes;
  cprank = t->cprank;

  xfibers0 = ft->xfibers[mode][0];
  xfibers1 = ft->xfibers[mode][1];

  cnt = ft->topmostcnt[mode];
	
  mat1 = t->mat[ft->order[mode][1]];
  mat2 = t->mat[ft->order[mode][2]];

  if(mode == ft->lmode)
    {
      fibers = ft->slfibers;
      vals = ft->slvals;
    }
  else
    {
      fibers = ft->lfibers;
      vals = ft->lvals;
    }

  acc = (double *)myMalloc(sizeof(double)*cprank);

  clock_t start, end;

  MPI_Barrier(MPI_COMM_WORLD);
  //time = (double)get_wc_time();
  start = clock();

  int fc = 0;
  for(it = 0; it < niters; it++)
    {

      for(i0 = 0; i0 < cnt; i0++)
        {
          base0 = i0*2;
          st0 = xfibers0[base0];
          en0 = xfibers0[base0+2];
          base0 = xfibers0[base0+1]*cprank;

          for(i1 = st0; i1 < en0; i1++)
            {
              base1 = i1*2;
              st1 = xfibers1[base1];
              en1 = xfibers1[base1+2];
              base1 = xfibers1[base1+1]*cprank;

              memset(acc, 0, sizeof(double)*cprank);
			  fc++;
              for(i2 = st1; i2 < en1; i2++)
                {
                  base2 = fibers[i2]*cprank; 
                  for(j = 0; j < cprank; j++)
                    acc[j] += mat2[base2+j]*vals[i2];
                }
			
              for(j = 0; j < cprank; j++)
                matm[base0+j] += acc[j]*mat1[base1+j];
            }

        }
    }

  MPI_Barrier(MPI_COMM_WORLD);
  //time = ((double)get_wc_time() - time)/niters;
  end = clock();
  time = (double)(end-start)/(CLOCKS_PER_SEC/1000);
  time /= niters;

  fc /= niters;
  fc += t->nnz;

  cnt += fc;
  
  int avg, max, avgs, maxs;
  MPI_Reduce(&fc, &avg, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
  MPI_Reduce(&fc, &max, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);

  MPI_Reduce(&cnt, &avgs, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
  MPI_Reduce(&cnt, &maxs, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);

  cnt_st[mode*4+0] = max;
  cnt_st[mode*4+1] = avg/t->npes;
  cnt_st[mode*4+2] = maxs;
  cnt_st[mode*4+3] = avgs/t->npes;

  free(acc);

  return time;
}

double mttkrp_fiber_4_stat(struct tensor *t, struct fibertensor *ft, int mode, double *matm, int niters, int *cnt_st)
{
  int it, j, nmodes, cprank, cnt, *xfibers0, *xfibers1, *xfibers2, *fibers, i0, st0, en0, i1, st1, en1, i2, st2, en2, en3, i3, base0, base1, base2, base3;
  double *mat1, *mat2, *mat3, *vals, *acc, *acc2, time;

  nmodes = t->nmodes;
  cprank = t->cprank;

  xfibers0 = ft->xfibers[mode][0];
  xfibers1 = ft->xfibers[mode][1];
  xfibers2 = ft->xfibers[mode][2];

  cnt = ft->topmostcnt[mode];
	
  mat1 = t->mat[ft->order[mode][1]];
  mat2 = t->mat[ft->order[mode][2]];
  mat3 = t->mat[ft->order[mode][3]];

  if(mode == ft->lmode)
    {
      fibers = ft->slfibers;
      vals = ft->slvals;
    }
  else
    {
      fibers = ft->lfibers;
      vals = ft->lvals;
    }

  acc = (double *)myMalloc(sizeof(double)*cprank);
  acc2 = (double *)myMalloc(sizeof(double)*cprank); 

  clock_t start, end;
	
  MPI_Barrier(MPI_COMM_WORLD);
  //time = (double)get_wc_time();
  start = clock();

  int fc = 0, sc = 0;
  for(it = 0; it < niters; it++)
    {

      for(i0 = 0; i0 < cnt; i0++)
        {
          base0 = i0*2;
          st0 = xfibers0[base0];
          en0 = xfibers0[base0+2];
          base0 = xfibers0[base0+1]*cprank;

          for(i1 = st0; i1 < en0; i1++)
            {
              base1 = i1*2;
              st1 = xfibers1[base1];
              en1 = xfibers1[base1+2];
              base1 = xfibers1[base1+1]*cprank;

              memset(acc, 0, sizeof(double)*cprank);
			  sc++;
              for(i2 = st1; i2 < en1; i2++)
                {
                  base2 = i2*2;
                  st2 = xfibers2[base2];
                  en2 = xfibers2[base2+2];
                  base2 = xfibers2[base2+1]*cprank;

                  memset(acc2, 0, sizeof(double)*cprank);
				  fc++;
                  for(i3 = st2; i3 < en2; i3++)
                    {
                      base3 = fibers[i3]*cprank;
                      for(j = 0; j < cprank; j++)
                        acc2[j] += mat3[base3+j]*vals[i3];
                    }
				
                  for(j = 0; j < cprank; j++)
                    acc[j] += acc2[j]*mat2[base2+j];
				

                }
              for(j = 0; j < cprank; j++)
                matm[base0+j] += acc[j]*mat1[base1+j];
			
            }

        }
    }

  MPI_Barrier(MPI_COMM_WORLD);
  //time = ((double)get_wc_time() - time)/niters;
  end = clock();
  time = (double)(end-start)/(CLOCKS_PER_SEC/1000);
  time /= niters;

  fc /= niters;
  sc /= niters;

  fc += t->nnz;
  sc += fc;
  
  int avg, max, avgs, maxs;
  MPI_Reduce(&fc, &avg, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
  MPI_Reduce(&fc, &max, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);

  MPI_Reduce(&sc, &avgs, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
  MPI_Reduce(&sc, &maxs, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);

  cnt_st[mode*4+0] = max;
  cnt_st[mode*4+1] = avg/t->npes;
  cnt_st[mode*4+2] = maxs;
  cnt_st[mode*4+3] = avgs/t->npes;
  
  free(acc);
  free(acc2);

  return time;
}


void mttkrp(struct tensor *t, struct fibertensor *ft, int mode, double *matm)
{
  if(t->fiber)
    {
      if(t->nmodes == 3)
        mttkrp_fiber_3(t, ft, mode, matm);
      else if(t->nmodes == 4)
        mttkrp_fiber_4(t, ft, mode, matm);
      else{
        printf("Not yet\n");
		exit(1);
	  }


    }
  else
    mttkrp_nnz(t, mode, matm);
		
}

double mttkrp_stat(struct tensor *t, struct fibertensor *ft, int mode, double *matm, int niters, int *cnt_st)
{
  if(t->fiber)
    {
      if(t->nmodes == 3)
		  return mttkrp_fiber_3_stat(t, ft, mode, matm, niters, cnt_st);
      else if(t->nmodes == 4)
		  return mttkrp_fiber_4_stat(t, ft, mode, matm, niters, cnt_st);
      else
        {
          printf("Not yet\n");
          return -1;
        }
    }
  else
    return mttkrp_nnz_stat(t, mode, matm, niters);
		
}


void compute_inverse(struct tensor *t, int mode, double *inverse)
{
  int i, j, k, nmodes, size, base, cprank;
  double *uTu, inner;

  cprank = t->cprank;
  size = cprank*cprank;
  for(i = 0; i < size; i++)
    inverse[i] = 1.0;
		

  nmodes = t->nmodes;
  uTu = t->uTu;
  for(i = 0; i < nmodes; i++) 
    {
      if(i != mode)
        {
          base = size*i;
          for(j = 0; j < size; j++) //CAN BE HALVED - SYM
            inverse[j] *= uTu[base+j]; 
        }
    }

  //Cholesky factorization
  double *lmatrix = (double *)myMalloc(sizeof(double)*size);
  setdoublezero(lmatrix, size);

  for(i = 0; i < cprank; i++)
    for(j = 0; j <= i; j++)
      {
        inner = 0.0;
        for(k = 0; k < j; k++)
          inner += lmatrix[i*cprank+k]*lmatrix[j*cprank+k];
        if(i == j)
          lmatrix[i*cprank+j] = sqrt(inverse[i*cprank+i]-inner);
        else
          lmatrix[i*cprank+j] = (inverse[i*cprank+j] - inner) / lmatrix[j*cprank+j];

      }

  //identity
  for(i = 0; i < size; i++)
    inverse[i] = 0.0;
 
  for(i = 0; i < cprank; i++)
    inverse[i*cprank+i] = 1.0;

  //forward solve
	
  /*	for(i = 0; i < cprank; i++)
        for(j = 0; j < cprank; j++)
        {
        inner = 0.0;
        for(k = 0; k < i; k++)
        inner += lmatrix[i*cprank+k]*inverse[k*cprank+j];
        inverse[i*cprank+j] = (inverse[i*cprank+j] - inner) / lmatrix[i*cprank+i];
        }
  */
  for(i = 0; i < cprank; i++)
    inverse[i] /= lmatrix[0];

  for(i = 1; i < cprank; i++)
    {	for(j = 0; j < i; j++)
        for(k = 0; k < cprank; k++)
          inverse[i*cprank+k] -= lmatrix[i*cprank+j]*inverse[j*cprank+k]; 

      for(k = 0; k < cprank; k++)
        inverse[i*cprank+k] /= lmatrix[i*cprank+i];
    }

  //transpose
  for(i = 0; i < cprank; i++)
    for(j = i+1; j < cprank; j++)
      {
        lmatrix[i*cprank+j] = lmatrix[j*cprank+i];
        lmatrix[j*cprank+i] = 0.0;
      }


  //backward
  /*	for(i = cprank-1; i >= 0; i--)
        for(j = 0; j < cprank; j++)
        {
        inner = 0.0;
        for(k = i+1; k < cprank; k++)
        inner += lmatrix[i*cprank+k]*inverse[k*cprank+j];
        inverse[i*cprank+j] = (inverse[i*cprank+j] - inner) / lmatrix[i*cprank+i];
        }
  */
	
  i = cprank-1;
  for(k = 0; k < cprank; k++)
    inverse[i*cprank+k] /= lmatrix[i*cprank+i];

  int row;
  for(row = 2; row <= cprank; row++)
    {
      i = cprank - row;
      for(j = i+1; j < cprank; j++)
        for(k = 0; k < cprank; k++)
          inverse[i*cprank+k] -= lmatrix[i*cprank+j]*inverse[j*cprank+k];

      for(k = 0; k < cprank; k++)
        inverse[i*cprank+k] /= lmatrix[i*cprank+i];
    }

  free(lmatrix);

}
 

int example()
{
  double *A, *B, *C;
  int m, n, k, i, j;
  double alpha, beta;

  printf ("\n This example computes real matrix C=alpha*A*B+beta*C using \n"
          " Intel(R) MKL function dgemm, where A, B, and  C are matrices and \n"
          " alpha and beta are double precision scalars\n\n");

  m = 5, k = 2, n = 3;
  printf (" Initializing data for matrix multiplication C=A*B for matrix \n"
          " A(%ix%i) and matrix B(%ix%i)\n\n", m, k, k, n);
  alpha = 1.0; beta = 0.0;

  printf (" Allocating memory for matrices aligned on 64-byte boundary for better \n"
          " performance \n\n");
  A = (double *)malloc( m*k*sizeof( double ));
  B = (double *)malloc( m*n*sizeof( double ));
  C = (double *)malloc( k*n*sizeof( double ));
  if (A == NULL || B == NULL || C == NULL) {
    printf( "\n ERROR: Can't allocate memory for matrices. Aborting... \n\n");
    free(A);
    free(B);
    free(C);
    return 1;
  }

  printf (" Intializing matrix data \n\n");
  for (i = 0; i < (m*k); i++) {
    A[i] = (double)(i+1);
  }

  for (i = 0; i < (m*n); i++) {
    B[i] = (double)(-i-1);
  }

  for (i = 0; i < (k*n); i++) {
    C[i] = 0.0;
  }

  printf (" Computing matrix product using Intel(R) MKL dgemm function via CBLAS interface \n\n");
  cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, 
              k, n, m, alpha, A, k, B, n, beta, C, n);
  printf ("\n Computations completed.\n\n");

  printf (" Top left corner of matrix A: \n");
  for (i=0; i<min(m,6); i++) {
    for (j=0; j<min(k,6); j++) {
      printf ("%12.0f", A[j+i*k]);
    }
    printf ("\n");
  }

  printf ("\n Top left corner of matrix B: \n");
  for (i=0; i<min(m,6); i++) {
    for (j=0; j<min(n,6); j++) {
      printf ("%12.0f", B[j+i*n]);
    }
    printf ("\n");
  }
    
  printf ("\n Top left corner of matrix C: \n");
  for (i=0; i<min(k,6); i++) {
    for (j=0; j<min(n,6); j++) {
      printf ("%12.5G", C[j+i*n]);
    }
    printf ("\n");
  }

  printf ("\n Deallocating memory \n\n");
  free(A);
  free(B);
  free(C);

  printf (" Example completed. \n\n");
  return 0;
}


void matrix_multiply(struct tensor *t, int mode, double *matm, double *inverse)
{
  int i, j, k, cprank, dim;
  double *matw, inner, alpha, beta;
	
  matw = t->mat[mode];
  dim = t->ldims[mode];
  cprank = t->cprank;

  /* 	for(i = 0; i < dim; i++) */
  /* 	{ */
  /* 		for(j = 0; j < cprank; j++) */
  /* 		{ */
  /* 			inner = 0; */
  /* 			for(k = 0; k < cprank; k++) */
  /* 				inner += matm[i*cprank+k]*inverse[k*cprank+j]; */
  /* 			matw[i*cprank+j] = inner; */
  /* 		} */
  /* 	} */

  alpha = 1.0;
  beta = 0.0;
  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
              dim, cprank, cprank, alpha, matm, cprank, inverse, cprank, beta, matw, cprank);
 
}

void normalize2(struct tensor *t, int mode, double *lambda)
{
  int i, j, dim, cprank, ptr;
  double *mat, v, w, *locallambda;


  cprank = t->cprank;
  mat = t->mat[mode];
  dim = t->ldims[mode];

  locallambda = (double *)myMalloc(cprank*sizeof(double));
  setdoublezero(locallambda, cprank);

  for(i = 0; i < cprank; i++)
    {

      locallambda[i]=cblas_ddot(dim, &mat[i], cprank, &mat[i], cprank);
      /* 		v = 0; */
      /* 		for(j = 0; j < dim; j++) */
      /* 		{ */
      /* 			w = mat[j*cprank+i]; */
      /* 			v += w*w; */
      /* 		} */
      /* 		locallambda[i] = v; */

    }


  MPI_Allreduce(locallambda, lambda, cprank, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

  for(i = 0; i < cprank; i++)
    lambda[i] = sqrt(lambda[i]);


  ptr = 0;
  for(i = 0; i < dim; i++)
    for(j = 0; j < cprank; j++)
      mat[ptr++] /= lambda[j];

  free(locallambda);

		
}

void normalizemax(struct tensor *t, int mode, double *lambda)
{
  int i, j, dim, cprank, ptr;
  double *mat, w, *locallambda;


  cprank = t->cprank;
  mat = t->mat[mode];
  dim = t->ldims[mode];

  locallambda = (double *)myMalloc(cprank*sizeof(double));
  setdoublezero(locallambda, cprank);
	
  ptr = 0;
  for(i = 0; i < dim; i++)
    {
      for(j = 0; j < cprank; j++)
        {
          w = mat[ptr++];
          if(locallambda[j] < w)
            locallambda[j] = w;
        }
    }


  MPI_Allreduce(locallambda, lambda, cprank, MPI_DOUBLE, MPI_MAX,  MPI_COMM_WORLD);

  for(i = 0; i < cprank; i++)
    if(lambda[i] < 1.0)
      lambda[i] = 1.0;
  ptr = 0;
  for(i = 0; i < dim; i++)
    for(j = 0; j < cprank; j++)
      mat[ptr++] /= lambda[j];

  free(locallambda);
}

double receive_partial_products_stat(struct tensor *t, int mode, double *matm, int niters)
{
  double time;
	
  time = 0.0;

  if(t->layersize[mode] > 1)
    {
      
      int i, it, j, iwrite, iread, nrecvwho, nsendwho, *recvwho, *sendwho, *xrecvind, *xsendind, *recvind, who, cprank, layersize, *sendcnt, *recvcnt, *senddisp, *recvdisp;
      double *recvbuf;
      struct comm *co;
      MPI_Comm layercomm;

      layersize = t->layersize[mode];
      layercomm = t->layercomm[mode];
      cprank = t->cprank;
      co = t->comm;

      nrecvwho = co->nsendwho[mode];
      recvwho = co->sendwho[mode];
      xrecvind = co->xsendind[mode];

      nsendwho = co->nrecvwho[mode];
      sendwho = co->recvwho[mode];
      xsendind = co->xrecvind[mode];

      recvind = co->sendind[mode];
      recvbuf = co->buffer;

      if(t->alltoall)
        {
          sendcnt = (int *)myMalloc(layersize*sizeof(int));
          setintzero(sendcnt, layersize);
          recvcnt = (int *)myMalloc(layersize*sizeof(int));
          setintzero(recvcnt, layersize);
          senddisp = (int *)myMalloc(layersize*sizeof(int));
          setintzero(senddisp, layersize);
          recvdisp = (int *)myMalloc(layersize*sizeof(int));
          setintzero(recvdisp, layersize);

          for(i = 0; i < nsendwho; i++)
            {
              who = sendwho[i];
              sendcnt[who] = (xsendind[who+1]-xsendind[who])*cprank;
              senddisp[who] = xsendind[who]*cprank;
            }
		
          for(i = 0; i < nrecvwho; i++)
            {
              who = recvwho[i];
              recvcnt[who] = (xrecvind[who+1]-xrecvind[who])*cprank;
              recvdisp[who] = xrecvind[who]*cprank;
            }

          int totalsend = 0, totalrecv = 0;
          for(i = 0; i < layersize; i++)
            {
              totalsend += sendcnt[i];
              totalrecv += recvcnt[i];
            }
			
          clock_t start, end;

          MPI_Barrier(MPI_COMM_WORLD);
          //start = MPI_Wtime();
          start = clock(); 
			
          for(it = 0; it < niters; it++)
            MPI_Alltoallv(matm, sendcnt, senddisp, MPI_DOUBLE, recvbuf, recvcnt, recvdisp, MPI_DOUBLE, layercomm);
			
          MPI_Barrier(MPI_COMM_WORLD);
          end = clock();
          time = (double) (end - start)/(CLOCKS_PER_SEC/1000);
          time /= niters;

          free(sendcnt);
          free(recvcnt);
          free(senddisp);
          free(recvdisp);

        }
      else
        {
          MPI_Request req[nrecvwho];
          MPI_Status sta[nrecvwho];

          clock_t start, end;

          MPI_Barrier(MPI_COMM_WORLD);
          //time = (double) get_wc_time();
          //time = MPI_Wtime();
          start = clock();

          for(it = 0; it < niters; it++)
            {
              for(i = 0; i < nrecvwho; i++)
                {
                  who = recvwho[i];
                  MPI_Irecv(&recvbuf[xrecvind[who]*cprank], (xrecvind[who+1]-xrecvind[who])*cprank, MPI_DOUBLE, who, 3, layercomm, &req[i]);
                }
              for(i = 0; i < nsendwho; i++)
                {
                  who = sendwho[i];
                  MPI_Send(&matm[xsendind[who]*cprank], (xsendind[who+1]-xsendind[who])*cprank, MPI_DOUBLE, who, 3, layercomm);
                }
              MPI_Waitall(nrecvwho, req, sta);
            }

          MPI_Barrier(MPI_COMM_WORLD);
          //time = ((double) get_wc_time() - time)/1000;
          //time = MPI_Wtime() - time; 

          end = clock();
          time = (double) (end - start)/(CLOCKS_PER_SEC/1000);
          time /= niters;

        }
    }
  return time;
}

double receive_partial_products_stat_fg(struct tensor *t, int mode, double *matm, int niters)
{
  double time;
	
  time = 0.0;
      
  int i, it, j, iwrite, iread, nrecvwho, nsendwho, *recvwho, *sendwho, *xrecvind, *xsendind, *recvind, who, cprank, size, *sendcnt, *recvcnt, *senddisp, *recvdisp;
  double *recvbuf;
  struct comm *co;

  size = t->npes;
  cprank = t->cprank;
  co = t->comm;

  nrecvwho = co->nsendwho[mode];
  recvwho = co->sendwho[mode];
  xrecvind = co->xsendind[mode];

  nsendwho = co->nrecvwho[mode];
  sendwho = co->recvwho[mode];
  xsendind = co->xrecvind[mode];

  recvind = co->sendind[mode];
  recvbuf = co->buffer;

  if(t->alltoall)
    {
      sendcnt = (int *)myMalloc(size*sizeof(int));
      setintzero(sendcnt, size);
      recvcnt = (int *)myMalloc(size*sizeof(int));
      setintzero(recvcnt, size);
      senddisp = (int *)myMalloc(size*sizeof(int));
      setintzero(senddisp, size);
      recvdisp = (int *)myMalloc(size*sizeof(int));
      setintzero(recvdisp, size);

      for(i = 0; i < nsendwho; i++)
        {
          who = sendwho[i];
          sendcnt[who] = (xsendind[who+1]-xsendind[who])*cprank;
          senddisp[who] = xsendind[who]*cprank;
        }
		
      for(i = 0; i < nrecvwho; i++)
        {
          who = recvwho[i];
          recvcnt[who] = (xrecvind[who+1]-xrecvind[who])*cprank;
          recvdisp[who] = xrecvind[who]*cprank;
        }

      int totalsend = 0, totalrecv = 0;
      for(i = 0; i < size; i++)
        {
          totalsend += sendcnt[i];
          totalrecv += recvcnt[i];
        }
			
      clock_t start, end;

      MPI_Barrier(MPI_COMM_WORLD);
      //start = MPI_Wtime();
      start = clock(); 
			
      for(it = 0; it < niters; it++)
        MPI_Alltoallv(matm, sendcnt, senddisp, MPI_DOUBLE, recvbuf, recvcnt, recvdisp, MPI_DOUBLE, MPI_COMM_WORLD);
			
      MPI_Barrier(MPI_COMM_WORLD);
      end = clock();
      time = (double) (end - start)/(CLOCKS_PER_SEC/1000);
      time /= niters;

      free(sendcnt);
      free(recvcnt);
      free(senddisp);
      free(recvdisp);

    }
  else
    {
      MPI_Request req[nrecvwho];
      MPI_Status sta[nrecvwho];

      clock_t start, end;

      MPI_Barrier(MPI_COMM_WORLD);
      //time = (double) get_wc_time();
      //time = MPI_Wtime();
      start = clock();

      for(it = 0; it < niters; it++)
        {
          for(i = 0; i < nrecvwho; i++)
            {
              who = recvwho[i];
              MPI_Irecv(&recvbuf[xrecvind[who]*cprank], (xrecvind[who+1]-xrecvind[who])*cprank, MPI_DOUBLE, who, 3, MPI_COMM_WORLD, &req[i]);
            }
          for(i = 0; i < nsendwho; i++)
            {
              who = sendwho[i];
              MPI_Send(&matm[xsendind[who]*cprank], (xsendind[who+1]-xsendind[who])*cprank, MPI_DOUBLE, who, 3, MPI_COMM_WORLD);
            }
          MPI_Waitall(nrecvwho, req, sta);
        }

      MPI_Barrier(MPI_COMM_WORLD);
      //time = ((double) get_wc_time() - time)/1000;
      //time = MPI_Wtime() - time; 

      end = clock();
      time = (double) (end - start)/(CLOCKS_PER_SEC/1000);
      time /= niters;

    }

  return time;
}



void receive_partial_products(struct tensor *t, int mode, double *matm)
{
  if(t->layersize[mode] > 1)
    {
		
      int i, j, iwrite, iread, nrecvwho, nsendwho, *recvwho, *sendwho, *xrecvind, *xsendind, *recvind, who, cprank, layersize, *sendcnt, *recvcnt, *senddisp, *recvdisp;
      double *recvbuf;
      struct comm *co;
      MPI_Comm layercomm;

      layersize = t->layersize[mode];
      layercomm = t->layercomm[mode];
      cprank = t->cprank;
      co = t->comm;

      nrecvwho = co->nsendwho[mode];
      recvwho = co->sendwho[mode];
      xrecvind = co->xsendind[mode];

      nsendwho = co->nrecvwho[mode];
      sendwho = co->recvwho[mode];
      xsendind = co->xrecvind[mode];

      recvind = co->sendind[mode];
      recvbuf = co->buffer;

      if(t->alltoall)
        {
          sendcnt = (int *)myMalloc(layersize*sizeof(int));
          setintzero(sendcnt, layersize);
          recvcnt = (int *)myMalloc(layersize*sizeof(int));
          setintzero(recvcnt, layersize);
          senddisp = (int *)myMalloc(layersize*sizeof(int));
          setintzero(senddisp, layersize);
          recvdisp = (int *)myMalloc(layersize*sizeof(int));
          setintzero(recvdisp, layersize);
          
          for(i = 0; i < nsendwho; i++)
            {
              who = sendwho[i];
              sendcnt[who] = (xsendind[who+1]-xsendind[who])*cprank;
              senddisp[who] = xsendind[who]*cprank;
            }
		
          for(i = 0; i < nrecvwho; i++)
            {
              who = recvwho[i];
              recvcnt[who] = (xrecvind[who+1]-xrecvind[who])*cprank;
              recvdisp[who] = xrecvind[who]*cprank;
            }

          MPI_Alltoallv(matm, sendcnt, senddisp, MPI_DOUBLE, recvbuf, recvcnt, recvdisp, MPI_DOUBLE, layercomm);

          free(sendcnt);
          free(recvcnt);
          free(senddisp);
          free(recvdisp);

        }
      else
        {
          MPI_Request req[nrecvwho];

          for(i = 0; i < nrecvwho; i++)
            {
              who = recvwho[i];
              MPI_Irecv(&recvbuf[xrecvind[who]*cprank], (xrecvind[who+1]-xrecvind[who])*cprank, MPI_DOUBLE, who, 3, layercomm, &req[i]);
            }
          for(i = 0; i < nsendwho; i++)
            {
              who = sendwho[i];
              MPI_Send(&matm[xsendind[who]*cprank], (xsendind[who+1]-xsendind[who])*cprank, MPI_DOUBLE, who, 3, layercomm);
            }

          MPI_Status sta[nrecvwho];
          MPI_Waitall(nrecvwho, req, sta);
	
        }

      iread = 0;
      for(i = 0; i < xrecvind[layersize]; i++)
        {
          iwrite = recvind[i]*cprank;
          for(j = 0; j < cprank; j++)
            matm[iwrite++] +=  recvbuf[iread++];
        }
    }

}

void receive_partial_products_fg(struct tensor *t, int mode, double *matm)
{
		
  int i, j, iwrite, iread, nrecvwho, nsendwho, *recvwho, *sendwho, *xrecvind, *xsendind, *recvind, who, cprank, size, *sendcnt, *recvcnt, *senddisp, *recvdisp;
  double *recvbuf;
  struct comm *co;

  size = t->npes;
  cprank = t->cprank;
  co = t->comm;

  nrecvwho = co->nsendwho[mode];
  recvwho = co->sendwho[mode];
  xrecvind = co->xsendind[mode];

  nsendwho = co->nrecvwho[mode];
  sendwho = co->recvwho[mode];
  xsendind = co->xrecvind[mode];

  recvind = co->sendind[mode];
  recvbuf = co->buffer;

  if(t->alltoall)
    {
      sendcnt = (int *)myMalloc(size*sizeof(int));
      setintzero(sendcnt, size);
      recvcnt = (int *)myMalloc(size*sizeof(int));
      setintzero(recvcnt, size);
      senddisp = (int *)myMalloc(size*sizeof(int));
      setintzero(senddisp, size);
      recvdisp = (int *)myMalloc(size*sizeof(int));
      setintzero(recvdisp, size);
          
      for(i = 0; i < nsendwho; i++)
        {
          who = sendwho[i];
          sendcnt[who] = (xsendind[who+1]-xsendind[who])*cprank;
          senddisp[who] = xsendind[who]*cprank;
        }
		
      for(i = 0; i < nrecvwho; i++)
        {
          who = recvwho[i];
          recvcnt[who] = (xrecvind[who+1]-xrecvind[who])*cprank;
          recvdisp[who] = xrecvind[who]*cprank;
        }

      MPI_Alltoallv(matm, sendcnt, senddisp, MPI_DOUBLE, recvbuf, recvcnt, recvdisp, MPI_DOUBLE, MPI_COMM_WORLD);

      free(sendcnt);
      free(recvcnt);
      free(senddisp);
      free(recvdisp);

    }
  else
    {
      MPI_Request req[nrecvwho];

      for(i = 0; i < nrecvwho; i++)
        {
          who = recvwho[i];
          MPI_Irecv(&recvbuf[xrecvind[who]*cprank], (xrecvind[who+1]-xrecvind[who])*cprank, MPI_DOUBLE, who, 3, MPI_COMM_WORLD, &req[i]);
        }
      for(i = 0; i < nsendwho; i++)
        {
          who = sendwho[i];
          MPI_Send(&matm[xsendind[who]*cprank], (xsendind[who+1]-xsendind[who])*cprank, MPI_DOUBLE, who, 3, MPI_COMM_WORLD);
        }

      MPI_Status sta[nrecvwho];
      MPI_Waitall(nrecvwho, req, sta);
	
    }

  iread = 0;
  for(i = 0; i < xrecvind[size]; i++)
    {
      iwrite = recvind[i]*cprank;
      for(j = 0; j < cprank; j++)
        matm[iwrite++] +=  recvbuf[iread++];
    }

}


void send_updated_rows(struct tensor *t, int mode)
{

  if(t->layersize[mode] > 1)
    {
      int i, j, ptr, who, nrecvwho, nsendwho, layersize, cprank, *recvwho, *sendwho, *xsendind, *xrecvind, *sendind, *recvind, *map, start, end, *sendcnt, *recvcnt, *senddisp, *recvdisp;
      double *sendbuf, *mat;
      struct comm *co;
      MPI_Comm layercomm;
	
      layersize = t->layersize[mode];
      co = t->comm;
      cprank = t->cprank;
      layercomm = t->layercomm[mode];
      mat = t->mat[mode];

      nrecvwho = co->nrecvwho[mode];
      nsendwho = co->nsendwho[mode];
      recvwho = co->recvwho[mode];
      sendwho = co->sendwho[mode];

      xrecvind = co->xrecvind[mode];
      xsendind = co->xsendind[mode];

      recvind = co->recvind[mode];
      sendind = co->sendind[mode];

      // send computed rows
      sendbuf = co->buffer;
      ptr = 0;
      for(i = 0; i < nsendwho; i++)
        {
          who = sendwho[i];
          start = xsendind[who];
          end = xsendind[who+1];
          for(j = start; j < end; j++)
            {
              memcpy(&sendbuf[ptr], &mat[sendind[j]], sizeof(double)*cprank);
              ptr += cprank;
            }
        }


      if(t->alltoall)
        {
          sendcnt = (int *)myMalloc(layersize*sizeof(int));
          setintzero(sendcnt, layersize);
          recvcnt = (int *)myMalloc(layersize*sizeof(int));
          setintzero(recvcnt, layersize);
          senddisp = (int *)myMalloc(layersize*sizeof(int));
          setintzero(senddisp, layersize);
          recvdisp = (int *)myMalloc(layersize*sizeof(int));
          setintzero(recvdisp, layersize);
			
          for(i = 0; i < nrecvwho; i++)
            {
              who = recvwho[i];
              recvcnt[who] = (xrecvind[who+1]-xrecvind[who])*cprank;
              recvdisp[who] = xrecvind[who]*cprank;
            }
		
          for(i = 0; i < nsendwho; i++)
            {
              who = sendwho[i];
              sendcnt[who] = (xsendind[who+1]-xsendind[who])*cprank;
              senddisp[who] = xsendind[who]*cprank;
            }


          MPI_Alltoallv(sendbuf, sendcnt, senddisp, MPI_DOUBLE, mat, recvcnt, recvdisp, MPI_DOUBLE, layercomm);

          free(sendcnt);
          free(recvcnt);
          free(senddisp);
          free(recvdisp);

        }
      else
        {
          // issue Irecvs
          MPI_Request req[nrecvwho];

          for(i = 0; i < nrecvwho; i++)
            {
              who = recvwho[i];
              MPI_Irecv(&mat[xrecvind[who]*cprank], (xrecvind[who+1]-xrecvind[who])*cprank, MPI_DOUBLE, who, 4, layercomm, &req[i]);
            }

          // send computed rows
          for(i = 0; i < nsendwho; i++)
            {
              who = sendwho[i];
              MPI_Send(&sendbuf[xsendind[who]*cprank], (xsendind[who+1]-xsendind[who])*cprank, MPI_DOUBLE, who, 4, layercomm);
            }

          MPI_Status sta[nrecvwho];
          MPI_Waitall(nrecvwho, req, sta);

        }
    }
}

void send_updated_rows_fg(struct tensor *t, int mode)
{

  int i, j, ptr, who, nrecvwho, nsendwho, size, cprank, *recvwho, *sendwho, *xsendind, *xrecvind, *sendind, *recvind, *map, start, end, *sendcnt, *recvcnt, *senddisp, *recvdisp;
  double *sendbuf, *mat;
  struct comm *co;
	
  size = t->npes;
  co = t->comm;
  cprank = t->cprank;

  mat = t->mat[mode];

  nrecvwho = co->nrecvwho[mode];
  nsendwho = co->nsendwho[mode];
  recvwho = co->recvwho[mode];
  sendwho = co->sendwho[mode];

  xrecvind = co->xrecvind[mode];
  xsendind = co->xsendind[mode];

  recvind = co->recvind[mode];
  sendind = co->sendind[mode];

  // send computed rows
  sendbuf = co->buffer;
  ptr = 0;
  for(i = 0; i < nsendwho; i++)
    {
      who = sendwho[i];
      start = xsendind[who];
      end = xsendind[who+1];
      for(j = start; j < end; j++)
        {
          memcpy(&sendbuf[ptr], &mat[sendind[j]], sizeof(double)*cprank);
          ptr += cprank;
        }
    }


  if(t->alltoall)
    {
      sendcnt = (int *)myMalloc(size*sizeof(int));
      setintzero(sendcnt, size);
      recvcnt = (int *)myMalloc(size*sizeof(int));
      setintzero(recvcnt, size);
      senddisp = (int *)myMalloc(size*sizeof(int));
      setintzero(senddisp, size);
      recvdisp = (int *)myMalloc(size*sizeof(int));
      setintzero(recvdisp, size);
			
      for(i = 0; i < nrecvwho; i++)
        {
          who = recvwho[i];
          recvcnt[who] = (xrecvind[who+1]-xrecvind[who])*cprank;
          recvdisp[who] = xrecvind[who]*cprank;
        }
		
      for(i = 0; i < nsendwho; i++)
        {
          who = sendwho[i];
          sendcnt[who] = (xsendind[who+1]-xsendind[who])*cprank;
          senddisp[who] = xsendind[who]*cprank;
        }


      MPI_Alltoallv(sendbuf, sendcnt, senddisp, MPI_DOUBLE, mat, recvcnt, recvdisp, MPI_DOUBLE, MPI_COMM_WORLD);

      free(sendcnt);
      free(recvcnt);
      free(senddisp);
      free(recvdisp);

    }
  else
    {
      // issue Irecvs
      MPI_Request req[nrecvwho];

      for(i = 0; i < nrecvwho; i++)
        {
          who = recvwho[i];
          MPI_Irecv(&mat[xrecvind[who]*cprank], (xrecvind[who+1]-xrecvind[who])*cprank, MPI_DOUBLE, who, 4, MPI_COMM_WORLD, &req[i]);
        }

      // send computed rows
      for(i = 0; i < nsendwho; i++)
        {
          who = sendwho[i];
          MPI_Send(&sendbuf[xsendind[who]*cprank], (xsendind[who+1]-xsendind[who])*cprank, MPI_DOUBLE, who, 4, MPI_COMM_WORLD);
        }

      MPI_Status sta[nrecvwho];
      MPI_Waitall(nrecvwho, req, sta);

    }
}


double send_updated_rows_stat(struct tensor *t, int mode, int niters)
{
  double time;
	
  time = 0.0;

  if(t->layersize[mode] > 1)
    {
      int i, it, j, ptr, who, nrecvwho, nsendwho, layersize, cprank, *recvwho, *sendwho, *xsendind, *xrecvind, *sendind, *recvind, *map, start, end, *sendcnt, *recvcnt, *senddisp, *recvdisp;
      double *sendbuf, *mat;
      struct comm *co;
      MPI_Comm layercomm;
	
      layersize = t->layersize[mode];
      co = t->comm;
      cprank = t->cprank;
      layercomm = t->layercomm[mode];
      mat = t->mat[mode];

      nrecvwho = co->nrecvwho[mode];
      nsendwho = co->nsendwho[mode];
      recvwho = co->recvwho[mode];
      sendwho = co->sendwho[mode];

      xrecvind = co->xrecvind[mode];
      xsendind = co->xsendind[mode];

      recvind = co->recvind[mode];
      sendind = co->sendind[mode];

      // send computed rows
      sendbuf = co->buffer;
      ptr = 0;
      for(i = 0; i < nsendwho; i++)
        {
          who = sendwho[i];
          start = xsendind[who];
          end = xsendind[who+1];
          for(j = start; j < end; j++)
            {
              memcpy(&sendbuf[ptr], &mat[sendind[j]], sizeof(double)*cprank);
              ptr += cprank;
            }
        }


      if(t->alltoall)
        {
          sendcnt = (int *)myMalloc(layersize*sizeof(int));
          setintzero(sendcnt, layersize);
          recvcnt = (int *)myMalloc(layersize*sizeof(int));
          setintzero(recvcnt, layersize);
          senddisp = (int *)myMalloc(layersize*sizeof(int));
          setintzero(senddisp, layersize);
          recvdisp = (int *)myMalloc(layersize*sizeof(int));
          setintzero(recvdisp, layersize);
			
          for(i = 0; i < nrecvwho; i++)
            {
              who = recvwho[i];
              recvcnt[who] = (xrecvind[who+1]-xrecvind[who])*cprank;
              recvdisp[who] = xrecvind[who]*cprank;
            }
		
          for(i = 0; i < nsendwho; i++)
            {
              who = sendwho[i];
              sendcnt[who] = (xsendind[who+1]-xsendind[who])*cprank;
              senddisp[who] = xsendind[who]*cprank;
            }

          clock_t start, end;
          MPI_Barrier(MPI_COMM_WORLD);
          //time = (double) get_wc_time();
          start = clock();
          for(it = 0; it < niters; it++)			  
            MPI_Alltoallv(sendbuf, sendcnt, senddisp, MPI_DOUBLE, mat, recvcnt, recvdisp, MPI_DOUBLE, layercomm);

          MPI_Barrier(MPI_COMM_WORLD);
          //time = ((double) get_wc_time() - time)/niters;
          end = clock();
          time = (double) (end - start)/(CLOCKS_PER_SEC/1000);
          time /= niters;

          free(sendcnt);
          free(recvcnt);
          free(senddisp);
          free(recvdisp);

        }
      else
        {
          // issue Irecvs
          MPI_Request req[nrecvwho];
          MPI_Status sta[nrecvwho];

          clock_t start, end;
			
          MPI_Barrier(MPI_COMM_WORLD);
          //time = (double) get_wc_time();
          start = clock();
          for(it = 0; it < niters; it++)
            {
              for(i = 0; i < nrecvwho; i++)
                {
                  who = recvwho[i];
                  MPI_Irecv(&mat[xrecvind[who]*cprank], (xrecvind[who+1]-xrecvind[who])*cprank, MPI_DOUBLE, who, 4, layercomm, &req[i]);
                }

              // send computed rows
              for(i = 0; i < nsendwho; i++)
                {
                  who = sendwho[i];
                  MPI_Send(&sendbuf[xsendind[who]*cprank], (xsendind[who+1]-xsendind[who])*cprank, MPI_DOUBLE, who, 4, layercomm);
                }

              MPI_Waitall(nrecvwho, req, sta);
            }

          MPI_Barrier(MPI_COMM_WORLD);
          //time = ((double) get_wc_time() - time)/niters;
          end = clock();
          time = (double) (end - start)/(CLOCKS_PER_SEC/1000);
          time /= niters;
        }
    }

  return time;
}

double send_updated_rows_stat_fg(struct tensor *t, int mode, int niters)
{
  double time;
	
  time = 0.0;

  int i, it, j, ptr, who, nrecvwho, nsendwho, size, cprank, *recvwho, *sendwho, *xsendind, *xrecvind, *sendind, *recvind, *map, start, end, *sendcnt, *recvcnt, *senddisp, *recvdisp;
  double *sendbuf, *mat;
  struct comm *co;
	
  size = t->npes;
  co = t->comm;
  cprank = t->cprank;
  mat = t->mat[mode];

  nrecvwho = co->nrecvwho[mode];
  nsendwho = co->nsendwho[mode];
  recvwho = co->recvwho[mode];
  sendwho = co->sendwho[mode];

  xrecvind = co->xrecvind[mode];
  xsendind = co->xsendind[mode];

  recvind = co->recvind[mode];
  sendind = co->sendind[mode];

  // send computed rows
  sendbuf = co->buffer;
  ptr = 0;
  for(i = 0; i < nsendwho; i++)
    {
      who = sendwho[i];
      start = xsendind[who];
      end = xsendind[who+1];
      for(j = start; j < end; j++)
        {
          memcpy(&sendbuf[ptr], &mat[sendind[j]], sizeof(double)*cprank);
          ptr += cprank;
        }
    }


  if(t->alltoall)
    {
      sendcnt = (int *)myMalloc(size*sizeof(int));
      setintzero(sendcnt, size);
      recvcnt = (int *)myMalloc(size*sizeof(int));
      setintzero(recvcnt, size);
      senddisp = (int *)myMalloc(size*sizeof(int));
      setintzero(senddisp, size);
      recvdisp = (int *)myMalloc(size*sizeof(int));
      setintzero(recvdisp, size);
			
      for(i = 0; i < nrecvwho; i++)
        {
          who = recvwho[i];
          recvcnt[who] = (xrecvind[who+1]-xrecvind[who])*cprank;
          recvdisp[who] = xrecvind[who]*cprank;
        }
		
      for(i = 0; i < nsendwho; i++)
        {
          who = sendwho[i];
          sendcnt[who] = (xsendind[who+1]-xsendind[who])*cprank;
          senddisp[who] = xsendind[who]*cprank;
        }

      clock_t start, end;
      MPI_Barrier(MPI_COMM_WORLD);
      //time = (double) get_wc_time();
      start = clock();
      for(it = 0; it < niters; it++)			  
        MPI_Alltoallv(sendbuf, sendcnt, senddisp, MPI_DOUBLE, mat, recvcnt, recvdisp, MPI_DOUBLE, MPI_COMM_WORLD);

      MPI_Barrier(MPI_COMM_WORLD);
      //time = ((double) get_wc_time() - time)/niters;
      end = clock();
      time = (double) (end - start)/(CLOCKS_PER_SEC/1000);
      time /= niters;

      free(sendcnt);
      free(recvcnt);
      free(senddisp);
      free(recvdisp);

    }
  else
    {
      // issue Irecvs
      MPI_Request req[nrecvwho];
      MPI_Status sta[nrecvwho];

      clock_t start, end;
			
      MPI_Barrier(MPI_COMM_WORLD);
      //time = (double) get_wc_time();
      start = clock();
      for(it = 0; it < niters; it++)
        {
          for(i = 0; i < nrecvwho; i++)
            {
              who = recvwho[i];
              MPI_Irecv(&mat[xrecvind[who]*cprank], (xrecvind[who+1]-xrecvind[who])*cprank, MPI_DOUBLE, who, 4, MPI_COMM_WORLD, &req[i]);
            }

          // send computed rows
          for(i = 0; i < nsendwho; i++)
            {
              who = sendwho[i];
              MPI_Send(&sendbuf[xsendind[who]*cprank], (xsendind[who+1]-xsendind[who])*cprank, MPI_DOUBLE, who, 4, MPI_COMM_WORLD);
            }

          MPI_Waitall(nrecvwho, req, sta);
        }

      MPI_Barrier(MPI_COMM_WORLD);
      //time = ((double) get_wc_time() - time)/niters;
      end = clock();
      time = (double) (end - start)/(CLOCKS_PER_SEC/1000);
      time /= niters;
    }

  return time;
}


void compute_aTa(struct tensor *t, int mode)
{
  int i, j, k, size, cprank, ldim, ptr1, ptr2;
  double *buffer, *mat, inner, alpha, beta;

  cprank = t->cprank;
  size = cprank*cprank;
  ldim = t->ldims[mode];
  mat = t->mat[mode];

  buffer = (double *)myMalloc(sizeof(double)*size);
  setdoublezero(buffer, size);

  /* 	for(i = 0; i < cprank; i++) */
  /* 		for(j = i; j < cprank; j++) */
  /* 		{ */
  /* 			inner = 0; */
  /* 			ptr1 = i; */
  /* 			ptr2 = j;      */
  /* 			for(k = 0; k < ldim; k++) */
  /* 			{ */
  /* 				inner += mat[ptr1]*mat[ptr2]; */
  /* 				ptr1 += cprank; */
  /* 				ptr2 += cprank; */
  /* 			} */

  /* 			buffer[i*cprank+j] = inner; */
  /* 			if(i != j) */
  /* 				buffer[j*cprank+i] = inner; */
  /* 		} */

  alpha = 1.0;
  beta = 0.0;
  cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
              cprank, cprank, ldim, alpha, mat, cprank, mat, cprank, beta, buffer, cprank);


  MPI_Allreduce(buffer, &(t->uTu[mode*size]), size, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
		
}

double compute_input_norm(struct tensor *t)
{
  int i, lnnz;
  double mynorm, norm, v, *vals; 

  mynorm = 0;

  vals = t->vals;
  lnnz = t->nnz;
  for(i = 0; i < lnnz; i++)
    {
      v = vals[i];
      mynorm += v*v;
    }
	
  MPI_Allreduce(&mynorm, &norm, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

  return norm;
}


double compute_fit(struct tensor *t, double *matm, double inputnorm, int mode, double *lambda)
{
  int i, j, ptr, nmodes, cprank, size, ldim;
  double decompnorm, myinner, inner, residual, *buffer, *uTu, *mat;
	
  ldim = t->ldims[mode];
  mat = t->mat[mode];
  uTu = t->uTu;
  nmodes = t->nmodes;
  cprank = t->cprank;
  size = cprank*cprank;

  //compute decomposition norm
  decompnorm = 0.0;
  buffer = (double *)myMalloc(sizeof(double)*size);
  for(i = 0; i < size; i++)
    buffer[i] = 1.0;

  ptr = 0;
  for(i = 0; i < nmodes; i++)
    for(j = 0; j < size; j++)
      buffer[j] *= uTu[ptr++]; 

  for(i = 0; i < cprank; i++)
    for(j = 0; j < cprank; j++)
      decompnorm += buffer[i*cprank+j]*lambda[i]*lambda[j];

  decompnorm = fabs(decompnorm);
	
  //compute inner product (of input tensor and decomposition)
  for(i = 0; i < cprank; i++)
    buffer[i] = 0.0;
	
  ptr = 0;
  for(i = 0; i < ldim; i++)
    for(j = 0; j < cprank; j++)
      buffer[j] += mat[ptr]*matm[ptr++];

  myinner = 0.0;
  for(i = 0; i < cprank; i++)
    myinner += buffer[i]*lambda[i];
	
  free(buffer);

  MPI_Allreduce(&myinner, &inner, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
	
  residual = sqrt(inputnorm + decompnorm - 2*inner);	

  return 1 - (residual/sqrt(inputnorm)); 
}

void print_stat(char tensorfile[], int *meshdims, struct stat *st, int nmodes, int mype, int npes, double cptime,double cptime2, double *mttkrptime, double *comm1time, double *comm2time, int gnnz, int ckbd, int *cnt_st,int iter)
{
  int i, max, modest[5], maxreduced[5], totreduced[5];
  double ltime[3], gtime[3], gcptime,gcptime2, comptime = 0, commtime = 0;
  int myvol = 0, mymsg = 0, myrows = 0, maxrows,  maxvol, totvol, maxmsg, totmsg;
	

  if(mype == 0)
    {
      char tname[1024];
      substring(tensorfile, tname);
      printf("%s", tname);
	    
      if(ckbd != 2)
        {
          printf("_%d", meshdims[0]);
          for(i = 1; i < nmodes; i++)
            printf("x%d", meshdims[i]);
        }
      printf("\t");	   
    }


  for(i = 0; i < nmodes; i++)
    {
      myvol += 2*(st->recvvol[i] + st->sendvol[i]);
      mymsg += 2*(st->recvmsg[i] + st->sendmsg[i]);
    }

  myrows = 0;
  for(i = 0; i < nmodes; i++)
    myrows += st->row[i];

  MPI_Reduce(&myvol, &maxvol, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);
  MPI_Reduce(&myvol, &totvol, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

  MPI_Reduce(&mymsg, &maxmsg, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);
  MPI_Reduce(&mymsg, &totmsg, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

  MPI_Reduce(&myrows, &maxrows, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);
	
  for(i = 0; i < nmodes; i++)
    {
      ltime[0] = mttkrptime[i];
      ltime[1] = comm1time[i];
      ltime[2] = comm2time[i];

      MPI_Reduce(ltime, gtime, 3, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
      if(mype == 0)
        {
          comptime += gtime[0];
          commtime += gtime[1] + gtime[2];
        }
    }

  MPI_Reduce(&cptime, &gcptime, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
  MPI_Reduce(&cptime2, &gcptime2, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
  MPI_Reduce(&(st->nnz), &max, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);

  if(mype == 0)
    printf("%lf\t%d\t%d\t%d\t%d\t%d\t%lf\t%d\t%lf\t%lf\t", (double)max/(gnnz/npes)-1, maxrows, maxvol, totvol/npes, maxmsg, totmsg/npes, comptime, iter, gcptime,gcptime2);

  if(mype == 0)
  {
	  for(i = 0; i < nmodes; i++)
		  printf("%d\t%d\t%d\t%d\t", cnt_st[i*4+0], cnt_st[i*4+1], cnt_st[i*4+2], cnt_st[i*4+3]);
	  
	  if(nmodes==3)
		  printf("na\tna\tna\tna\t");

  }
  
  if(mype == 0)
    printf("%d\t%d\t", max, gnnz/npes);

  for(i = 0; i < nmodes; i++)
    {
      modest[0] = st->recvvol[i];
      modest[1] = st->sendvol[i];
      modest[2] = st->recvmsg[i];
      modest[3] = st->sendmsg[i];
      modest[4] = st->row[i];

      MPI_Reduce(modest, maxreduced, 5, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);
      MPI_Reduce(modest, totreduced, 5, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

      if(mype == 0) 
        printf("%d\t%d\t%d\t%d\t%d\t%d\t%d\t", maxreduced[0], maxreduced[1], totreduced[0]/npes, maxreduced[2], maxreduced[3], totreduced[2]/npes, maxreduced[4]);

    }
	

  if(mype == 0){
	  if(nmodes==3)
		  printf("na\tna\tna\tna\tna\tna\tna\t");
    
  }

  for(i = 0; i < nmodes; i++)
    {
      ltime[0] = mttkrptime[i];
      ltime[1] = comm1time[i];
      ltime[2] = comm2time[i];
	
      MPI_Reduce(ltime, gtime, 3, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
      if(mype == 0)
        printf("%lf\t%lf\t%lf\t", gtime[0], gtime[1], gtime[2]);
	
    }
	if(mype == 0 && nmodes==3)
        printf("na\tna\tna\t", gtime[0], gtime[1], gtime[2]);

}


int cp_als_stat(struct tensor *t, struct fibertensor *ft, int niters, double *mttkrptime, double *comm1time, double *comm2time, int *cnt_st)
{
  
  int it, i, nmodes, maxldim;
  double *matm;
	
  nmodes = t->nmodes;

  maxldim = t->ldims[0];
  for(i = 1; i < nmodes; i++)
    if(t->ldims[i] > maxldim)
      maxldim = t->ldims[i];

  matm = (double *)myMalloc(maxldim*t->cprank*sizeof(double));

  for(i = 0; i < nmodes; i++)
    {
      comm1time[i] = receive_partial_products_stat(t, i, matm, niters);
      mttkrptime[i] = mttkrp_stat(t, ft, i, matm, niters, cnt_st);
      comm2time[i] = send_updated_rows_stat(t, i, niters);
    }
	  
  free(matm);
	
  return 0;
}

int cp_als_stat_fg(struct tensor *t, struct fibertensor *ft, int niters, double *mttkrptime, double *comm1time, double *comm2time, int *cnt_st)
{
  
  int it, i, nmodes, maxldim;
  double *matm;
	
  nmodes = t->nmodes;

  maxldim = t->ldims[0];
  for(i = 1; i < nmodes; i++)
    if(t->ldims[i] > maxldim)
      maxldim = t->ldims[i];

  matm = (double *)myMalloc(maxldim*t->cprank*sizeof(double));

  for(i = 0; i < nmodes; i++)
    {
      comm1time[i] = receive_partial_products_stat_fg(t, i, matm, niters);
      mttkrptime[i] = mttkrp_stat(t, ft, i, matm, niters, cnt_st);
      comm2time[i] = send_updated_rows_stat_fg(t, i, niters);
    }
	  
  free(matm);
	
  return 0;
}


int cp_als(struct tensor *t, struct fibertensor *ft, int niters, double *cptime)
{
  
  int it, i, nmodes, maxldim;
  double *matm, *lambda, inputnorm, fit, oldfit, *inverse;
	
  nmodes = t->nmodes;
  lambda = (double *)myMalloc(t->cprank*sizeof(double));
       
  inputnorm = compute_input_norm(t);

  maxldim = t->ldims[0];
  for(i = 1; i < nmodes; i++)
    if(t->ldims[i] > maxldim)
      maxldim = t->ldims[i];

  matm = (double *)myMalloc(maxldim*t->cprank*sizeof(double));
  inverse = (double *)myMalloc(sizeof(double)*t->cprank*t->cprank);

  clock_t start, end;

  MPI_Barrier(MPI_COMM_WORLD);
  //*cptime = (double)get_wc_time();
  start = clock();
  for(it = 0; it < niters; it++)
    {
      for(i = 0; i < nmodes; i++)
        {
          setdoublezero(matm, t->ldims[i]*t->cprank);

          mttkrp(t, ft, i, matm);

          receive_partial_products(t, i, matm);

          compute_inverse(t, i, inverse);

          matrix_multiply(t, i, matm, inverse);

          if(it == 0)
            normalize2(t, i, lambda);
          else
            normalizemax(t, i, lambda);

          send_updated_rows(t, i);

          compute_aTa(t, i);

        }
      fit = compute_fit(t, matm, inputnorm, nmodes-1, lambda);
    }

  MPI_Barrier(MPI_COMM_WORLD);
  //*cptime = (((double)get_wc_time() - *cptime)/1000)/niters;
  end = clock();
  *cptime = (double)(end-start)/(CLOCKS_PER_SEC/1000);
  *cptime /= niters;

  free(matm);
  free(inverse);
	
  return 0;
}


int cp_als_fg(struct tensor *t, struct fibertensor *ft, int niters, double *cptime, double *cptime2)
{
  
  int it, i, nmodes, maxldim;
  double *matm, *lambda, inputnorm, fit, oldfit, *inverse;
	
  nmodes = t->nmodes;
  lambda = (double *)myMalloc(t->cprank*sizeof(double));
       
  inputnorm = compute_input_norm(t);

  maxldim = t->ldims[0];
  for(i = 1; i < nmodes; i++)
    if(t->ldims[i] > maxldim)
      maxldim = t->ldims[i];

  matm = (double *)myMalloc(maxldim*t->cprank*sizeof(double));
  inverse = (double *)myMalloc(sizeof(double)*t->cprank*t->cprank);

  //clock_t start, end;
  double start2, end2;

  MPI_Barrier(MPI_COMM_WORLD);
  //*cptime = (double)get_wc_time();
  start2=MPI_Wtime();
  //start = clock();
  //high_resolution_clock::time_point expand_time_A_1 = std::chrono::high_resolution_clock::now();
  auto startX = std::chrono::high_resolution_clock::now();
  for(it = 0; it < niters; it++)
    {
      for(i = 0; i < nmodes; i++)
        {
          setdoublezero(matm, t->ldims[i]*t->cprank);

          mttkrp(t, ft, i, matm);

          receive_partial_products_fg(t, i, matm);

          compute_inverse(t, i, inverse);

          matrix_multiply(t, i, matm, inverse);

          if(it == 0)
            normalize2(t, i, lambda);
          else
            normalizemax(t, i, lambda);

          send_updated_rows_fg(t, i);

          compute_aTa(t, i);

        }
      fit = compute_fit(t, matm, inputnorm, nmodes-1, lambda);
    }

  MPI_Barrier(MPI_COMM_WORLD);
  //high_resolution_clock::time_point expand_time_A_2 = std::chrono::high_resolution_clock::now();
  auto endX = std::chrono::high_resolution_clock::now();
  
  
  

	


  
  
  
  
  
  
  
  //*cptime = (((double)get_wc_time() - *cptime)/1000)/niters;
  end2 = MPI_Wtime();
  //end = clock();
  //std::chrono::duration<double> diff = endX-startX;
  //std::chrono::duration<double> time_span_A = std::chrono::duration_cast<std::chrono::duration<double>>(expand_time_A_2 - expand_time_A_1);
	//std::chrono::duration<double> time_span_A = std::chrono::duration_cast<std::chrono::duration<double>>(expand_time_A_2 - expand_time_A_1);
	//*cptime = time_span_A.count();//, expand_time_A_global;

	std::chrono::duration<double> time_span_A = std::chrono::duration_cast<std::chrono::duration<double>>(endX-startX);
	*cptime = time_span_A.count();
	//std::cout << "Time to fill and iterate a vector of ints : " << time << " s\n";

  
  
  
  
  
  
  
  //*cptime = (double)(end-start)/(CLOCKS_PER_SEC/1000);
  *cptime2 = (double)(end2-start2);
  *cptime /= niters;
  *cptime2 /= niters;

  free(matm);
  free(inverse);
	
  return 0;
}


int cp_als_barrier(struct tensor *t, struct fibertensor *ft, int niters)
{
  
  int it, i, j, nmodes, maxldim;
  double *matm, *lambda, inputnorm, fit, oldfit, *inverse, time[9], totaltime;
	
  for(j = 0; j < 9; j++)
    time[j] = 0.0;

  nmodes = t->nmodes;
  lambda = (double *)myMalloc(t->cprank*sizeof(double));
       
  inputnorm = compute_input_norm(t);

  maxldim = t->ldims[0];
  for(i = 1; i < nmodes; i++)
    if(t->ldims[i] > maxldim)
      maxldim = t->ldims[i];

  matm = (double *)myMalloc(maxldim*t->cprank*sizeof(double));
  inverse = (double *)myMalloc(sizeof(double)*t->cprank*t->cprank);

  clock_t fstart, start, end;

  MPI_Barrier(MPI_COMM_WORLD);
  fstart = clock();
  for(it = 0; it < niters; it++)
    {
      for(i = 0; i < nmodes; i++)
        {
          MPI_Barrier(MPI_COMM_WORLD);
          start = clock();		
          setdoublezero(matm, t->ldims[i]*t->cprank);
          MPI_Barrier(MPI_COMM_WORLD);
          end = clock();
          time[0] += (double)(end-start);

          MPI_Barrier(MPI_COMM_WORLD);	      
          start = clock();		  
          mttkrp(t, ft, i, matm);	    
          MPI_Barrier(MPI_COMM_WORLD);
          end = clock();
          time[1] += (double)(end-start);
	      
          MPI_Barrier(MPI_COMM_WORLD);
          start = clock();
          receive_partial_products(t, i, matm);
          MPI_Barrier(MPI_COMM_WORLD);
          end = clock();
          time[2] += (double)(end-start);
	      
          MPI_Barrier(MPI_COMM_WORLD);
          start = clock();
          compute_inverse(t, i, inverse);		  
          MPI_Barrier(MPI_COMM_WORLD);
          end = clock();
          time[3] += (double)(end-start);
          
          MPI_Barrier(MPI_COMM_WORLD);
          start = clock();
          matrix_multiply(t, i, matm, inverse);
          MPI_Barrier(MPI_COMM_WORLD);
          end = clock();
          time[4] += (double)(end-start);

          MPI_Barrier(MPI_COMM_WORLD);
          start = clock();		  
          if(it == 0)
            normalize2(t, i, lambda);
          else
            normalizemax(t, i, lambda);		  
          MPI_Barrier(MPI_COMM_WORLD);
          end = clock();
          time[5] += (double)(end-start);
	      
          MPI_Barrier(MPI_COMM_WORLD);
          start = clock();
          send_updated_rows(t, i);
          MPI_Barrier(MPI_COMM_WORLD);
          end = clock();
          time[6] += (double)(end-start);
	      
          MPI_Barrier(MPI_COMM_WORLD);
          start = clock();
          compute_aTa(t, i);
          MPI_Barrier(MPI_COMM_WORLD);
          end = clock();
          time[7] += (double)(end-start);
        }
	  
      MPI_Barrier(MPI_COMM_WORLD);
      start = clock();
      fit = compute_fit(t, matm, inputnorm, nmodes-1, lambda);
      MPI_Barrier(MPI_COMM_WORLD);
      end = clock();
      time[8] += (double)(end-start);

    }

  MPI_Barrier(MPI_COMM_WORLD);
  end = clock();
  totaltime = (double)(end-fstart)/(CLOCKS_PER_SEC/1000);
  totaltime /= niters;

  if(t->mype == 0)
    {
      int y;
      for(y = 0; y < 9; y++)
        {
          time[y] /= (CLOCKS_PER_SEC/1000);
          time[y] /= niters;
          printf("(%d) %f\n", y, time[y]);
        }
      printf("(cp) %f\n", totaltime);
    }

  free(matm);
  free(inverse);
	
  return 0;
}

int cp_als_barrier_fg(struct tensor *t, struct fibertensor *ft, int niters)
{
  
  int it, i, j, nmodes, maxldim;
  double *matm, *lambda, inputnorm, fit, oldfit, *inverse, time[9], totaltime;
	
  for(j = 0; j < 9; j++)
    time[j] = 0.0;

  nmodes = t->nmodes;
  lambda = (double *)myMalloc(t->cprank*sizeof(double));
       
  inputnorm = compute_input_norm(t);

  maxldim = t->ldims[0];
  for(i = 1; i < nmodes; i++)
    if(t->ldims[i] > maxldim)
      maxldim = t->ldims[i];

  matm = (double *)myMalloc(maxldim*t->cprank*sizeof(double));
  inverse = (double *)myMalloc(sizeof(double)*t->cprank*t->cprank);

  clock_t fstart, start, end;

  MPI_Barrier(MPI_COMM_WORLD);
  fstart = clock();
  for(it = 0; it < niters; it++)
    {
      for(i = 0; i < nmodes; i++)
        {
          MPI_Barrier(MPI_COMM_WORLD);
          start = clock();		
          setdoublezero(matm, t->ldims[i]*t->cprank);
          MPI_Barrier(MPI_COMM_WORLD);
          end = clock();
          time[0] += (double)(end-start);

          MPI_Barrier(MPI_COMM_WORLD);	      
          start = clock();		  
          mttkrp(t, ft, i, matm);	    
          MPI_Barrier(MPI_COMM_WORLD);
          end = clock();
          time[1] += (double)(end-start);
	      
          MPI_Barrier(MPI_COMM_WORLD);
          start = clock();
          receive_partial_products_fg(t, i, matm);
          MPI_Barrier(MPI_COMM_WORLD);
          end = clock();
          time[2] += (double)(end-start);
	      
          MPI_Barrier(MPI_COMM_WORLD);
          start = clock();
          compute_inverse(t, i, inverse);		  
          MPI_Barrier(MPI_COMM_WORLD);
          end = clock();
          time[3] += (double)(end-start);
          
          MPI_Barrier(MPI_COMM_WORLD);
          start = clock();
          matrix_multiply(t, i, matm, inverse);
          MPI_Barrier(MPI_COMM_WORLD);
          end = clock();
          time[4] += (double)(end-start);

          MPI_Barrier(MPI_COMM_WORLD);
          start = clock();		  
          if(it == 0)
            normalize2(t, i, lambda);
          else
            normalizemax(t, i, lambda);		  
          MPI_Barrier(MPI_COMM_WORLD);
          end = clock();
          time[5] += (double)(end-start);
	      
          MPI_Barrier(MPI_COMM_WORLD);
          start = clock();
          send_updated_rows_fg(t, i);
          MPI_Barrier(MPI_COMM_WORLD);
          end = clock();
          time[6] += (double)(end-start);
	      
          MPI_Barrier(MPI_COMM_WORLD);
          start = clock();
          compute_aTa(t, i);
          MPI_Barrier(MPI_COMM_WORLD);
          end = clock();
          time[7] += (double)(end-start);
        }
	  
      MPI_Barrier(MPI_COMM_WORLD);
      start = clock();
      fit = compute_fit(t, matm, inputnorm, nmodes-1, lambda);
      MPI_Barrier(MPI_COMM_WORLD);
      end = clock();
      time[8] += (double)(end-start);

    }

  MPI_Barrier(MPI_COMM_WORLD);
  end = clock();
  totaltime = (double)(end-fstart)/(CLOCKS_PER_SEC/1000);
  totaltime /= niters;

  if(t->mype == 0)
    {
      int y;
      for(y = 0; y < 9; y++)
        {
          time[y] /= (CLOCKS_PER_SEC/1000);
          time[y] /= niters;
          printf("(%d) %f\n", y, time[y]);
        }
      printf("(cp) %f\n", totaltime);
    }

  free(matm);
  free(inverse);
	
  return 0;
}


void radixsort(int *inds, double *vals, int nnz, int nmodes, int *order, int *dims)
{

  int i, j, ptr, ptr2, *cnt, *tmpinds, size, mode, dim;
  double *tmpvals;
       
  size = nmodes*sizeof(int);

  tmpinds = (int *)myMalloc(nnz*nmodes*sizeof(int));
  tmpvals = (double *)myMalloc(nnz*sizeof(double));

  for(i = 0; i < nmodes; i++)
    {
      mode = order[nmodes-1-i];
      dim = dims[mode];

      cnt = (int *)myMalloc((dim+1)*sizeof(int));
      setintzero(cnt, dim+1);

      ptr = mode;
      for(j = 0; j < nnz; j++)
        {
          cnt[inds[ptr]+1]++;
          ptr += nmodes;
        }

      dim++;
		
      for(j = 2; j < dim; j++)
        cnt[j] += cnt[j-1];

      memcpy(tmpinds, inds, nnz*nmodes*sizeof(int));
      memcpy(tmpvals, vals, nnz*sizeof(double));

      ptr = mode;
      for(j = 0; j < nnz; j++)
        {	
          ptr2 = cnt[tmpinds[ptr]]++;
          memcpy(&inds[ptr2*nmodes], &tmpinds[j*nmodes], size);
          vals[ptr2] = tmpvals[j];
          ptr += nmodes;
        }

      free(cnt);

    }

  free(tmpinds);
  free(tmpvals);

}

void checksort(int *dims, int nnz, int nmodes, int *order)
{
  int i, j, cont;
  for(i = 1; i < nnz; i++)
    {
      cont = 1;
      for(j = 0; cont == 1 && j < nmodes; j++)
        if(dims[i*nmodes+order[j]] > dims[(i-1)*nmodes+order[j]])
          cont = 0;
        else if(dims[i*nmodes+order[j]] == dims[(i-1)*nmodes+order[j]])
          cont = 1;
        else
          {
            cont = 0;
            printf("adasda\n");
          }
    }
}


void get_longest_fibers(int *inds, double *vals, int nmodes, int longestmode, int nnz, struct fibertensor *ft)
{
  int i, *fibers, ptr;

  ft->lfibers = (int *)myMalloc(nnz*sizeof(int));
  fibers = ft->lfibers;

  ptr = longestmode;
  for(i = 0; i < nnz; i++)
    {	
      fibers[i] = inds[ptr];
      ptr += nmodes;
    }

  ft->lvals = (double *)myMalloc(nnz*sizeof(double));
  memcpy(ft->lvals, vals, nnz*sizeof(double));

}

void get_secondlongest_fibers(int *inds, double *vals, int nmodes, int secondlongestmode, int nnz, struct fibertensor *ft)
{
  int i, *fibers, ptr;

  ft->slfibers = (int *)myMalloc(nnz*sizeof(int));
  fibers = ft->slfibers;

  ptr = secondlongestmode;
  for(i = 0; i < nnz; i++)
    {	
      fibers[i] = inds[ptr];
      ptr += nmodes;
    }

  ft->slvals = (double *)myMalloc(nnz*sizeof(double));
  memcpy(ft->slvals, vals, nnz*sizeof(double));
}


void point_fibers(int mype, int *inds, int *order, int nmodes, int nnz, int mode, struct fibertensor *ft, int longest)
{
  int i, j, *shrdinds, last, nfib, **xfibers, *fibers, ptr, imode, cont, lastfibcnt, lastptd;
	

  ft->xfibers[mode] = (int **)myMalloc((nmodes-1)*sizeof(int *));
  shrdinds = (int *)myMalloc((nmodes-1)*sizeof(int));

  if(longest)
    fibers = ft->lfibers;
  else
    fibers = ft->slfibers;

  xfibers = ft->xfibers[mode];

  lastfibcnt = nnz;
  for(imode = 1; imode < nmodes; imode++)
    {
		
      //count the number of fibers
      last = 0;
      nfib = 0;
      while(last < lastfibcnt)
        {
          if(imode == 1)
            lastptd = last;
          else if(imode == 2)
            lastptd = xfibers[nmodes-imode][last*2];
          else
            {
              lastptd = xfibers[nmodes-imode][last*2];
              lastptd = xfibers[nmodes-imode+1][lastptd*2];
            }

          for(j = 0; j < nmodes-imode; j++)
            shrdinds[j] = inds[lastptd*nmodes + order[j]];
			
          cont = 1;
          i = last+1;
          while(cont && i < lastfibcnt) 
            {
              if(imode == 1)
                lastptd = i;
              else if(imode == 2)
                lastptd = xfibers[nmodes-imode][i*2];
              else
                {
                  lastptd = xfibers[nmodes-imode][i*2];
                  lastptd = xfibers[nmodes-imode+1][lastptd*2];
                }

              for(j = 0; j < nmodes-imode; j++)
                if(shrdinds[j] != inds[lastptd*nmodes+order[j]])
                  cont = 0;
              if(cont)
                  i++;
            }
			
          nfib++;
          last = i;	
        }
	
      //allocate space for the last level fiber pointer
      xfibers[nmodes-imode-1] = (int *)myMalloc((nfib+1)*2*sizeof(int));

      //find pointers to fiber starting positions
      //xfibers[nmodes-2][fid*2]: starting position of the fiber in fibers
      //xfibers[nmodes-2][fid*2+1]: the index of the fiber in the previous dimension
      last = 0;
      nfib = 0;
      xfibers[nmodes-imode-1][nfib*2] = last;
      while(last < lastfibcnt)
        {

          if(imode == 1)
            lastptd = last;
          else if(imode == 2)
            lastptd = xfibers[nmodes-imode][last*2];
          else
            {
              lastptd = xfibers[nmodes-imode][last*2];
              lastptd = xfibers[nmodes-imode+1][lastptd*2];
            }

          for(j = 0; j < nmodes-imode; j++)
            shrdinds[j] = inds[lastptd*nmodes + order[j]];
			
          xfibers[nmodes-imode-1][nfib*2+1] = shrdinds[nmodes-imode-1];
			
          cont = 1;
          i = last+1;
          while(cont && i < lastfibcnt) 
            {

              if(imode == 1)
                lastptd = i;
              else if(imode == 2)
                lastptd = xfibers[nmodes-imode][i*2];
              else
                {
                  lastptd = xfibers[nmodes-imode][i*2];
                  lastptd = xfibers[nmodes-imode+1][lastptd*2];
                }

              for(j = 0; j < nmodes-imode; j++)
                if(shrdinds[j] != inds[lastptd*nmodes+order[j]])
                  cont = 0;
              if(cont)
                  i++;
            }
			
          nfib++;
          last = i;
          xfibers[nmodes-imode-1][nfib*2] = last;
				
        }
      lastfibcnt = nfib;
    }
  ft->topmostcnt[mode] = nfib;

  free(shrdinds);

}


void get_sparsity_order(struct tensor *t, int *inds, int *order)
{
  int i, j, nmodes, nnz, *mark, *minecnt, **mine, ptr, ldim, *minei, cnt, min;

  nmodes = t->nmodes;
  nnz = t->nnz;

  mark = (int *)myMalloc(nmodes*sizeof(int));
  setintzero(mark, nmodes);
  minecnt = (int *)myMalloc(nmodes*sizeof(int));
  setintzero(minecnt, nmodes);
  mine = (int **)myMalloc(nmodes*sizeof(int *));

  for(i = 0; i < nmodes; i++)
    {
      mine[i] = (int *)myMalloc(t->ldims[i]*sizeof(int));
      setintzero(mine[i], t->ldims[i]);
    }

  ptr = 0;
  for(i = 0; i < nnz; i++)
    {
      for(j = 0; j < nmodes; j++)
        mine[j][inds[ptr++]] = 1;
    }

  for(i = 0; i < nmodes; i++)
    {
      cnt = 0;
      ldim = t->ldims[i];
      minei = mine[i];
      for(j = 0; j < ldim; j++)
        cnt += minei[j];

      minecnt[i] = cnt;
		
    }

  for(i = 0; i < nmodes; i++)
    {
      min = INT_MAX;
      for(j = 0; j < nmodes; j++)
        {
          if(mark[j] == 0 && minecnt[j] < min)
            {
              min = minecnt[j];
              order[i] = j;
            }
        }
      mark[order[i]] = 1;

    }

  for(i = 0; i < nmodes; i++)
    free(mine[i]);
  free(mine);
  free(minecnt);
  free(mark);

}

void get_fibertensor(struct tensor *t, struct fibertensor *ft)
{
  int nmodes, i, c, j, nnz, *inds, min, *order, lmode;

  nmodes = t->nmodes;
  nnz = t->nnz;

  t->sporder = (int *)myMalloc(nmodes*sizeof(int));
  get_sparsity_order(t, t->inds, t->sporder);
  ft->lmode = t->sporder[nmodes-1];

  ft->order = (int **)myMalloc(nmodes*sizeof(int *));
  ft->xfibers = (int ***)myMalloc(nmodes*sizeof(int **));
  ft->topmostcnt = (int *)myMalloc(nmodes*sizeof(int));

  for(i = 0; i < nmodes; i++)
    {

      ft->order[i] = (int *)myMalloc(nmodes*sizeof(int));
      order = ft->order[i];

      c = 0;
      order[c++] = i;
      for(j = 0; j < nmodes; j++)
        if(t->sporder[j] != i)
          order[c++] = t->sporder[j];

      radixsort(t->inds, t->vals, nnz, nmodes, order, t->ldims);
      checksort(t->inds, nnz, nmodes, order);

      if(i != ft->lmode)
        {
          if(ft->lfibers == NULL)
            get_longest_fibers(t->inds, t->vals, nmodes, ft->lmode, nnz, ft);
          point_fibers(t->mype, t->inds, order, nmodes, nnz, i, ft, 1);
        }
      else
        {
          if(ft->slfibers == NULL)
            get_secondlongest_fibers(t->inds, t->vals, nmodes, t->sporder[nmodes-2], nnz, ft);
          point_fibers(t->mype, t->inds, order, nmodes, nnz, i, ft, 0);
        }
    }

}



void read_ckbd_partition(char partfile[], struct tensor *t)
{
  int i, j, nmodes, npes, *meshdims, *gdims, dim, meshdim, mode, layersize;
  FILE *fpart;
  char tname[1024];


  fpart = fopen(partfile, "r");
  fscanf(fpart, "##%d %d", &nmodes, &npes);
  t->nmodes = nmodes;
  t->gdims = (int *)myMalloc(sizeof(int)*nmodes);

  if(npes != t->npes)
    {
      if(t->mype == 0)
        printf("The partition file is for %d processors!!\n", npes);
      MPI_Finalize();
      exit(1);
    }
  t->meshdims = (int *)myMalloc(sizeof(int)*nmodes);
  gdims = t->gdims;
  meshdims = t->meshdims;

  t->interpart = (int **)myMalloc(sizeof(int *)*nmodes);
  t->intrapart = (int **)myMalloc(sizeof(int *)*nmodes);


  for(i = 0; i < nmodes; i++)
    {
      fscanf(fpart, "\n##%d %d", &mode, &dim);
      gdims[mode] = dim;

      /* if(gdims[mode] != dim) */
      /*   { */
      /*     if(t->mype == 0) */
      /*       printf("The dimensions of tensor and partition don't match!!\n", npes); */
      /*     MPI_Finalize(); */
      /*     exit(1); */
      /*   } */

      fscanf(fpart, "\n#%d\n", &meshdim);
      meshdims[mode] = meshdim;
		
      t->interpart[mode] = (int *)myMalloc(sizeof(int)*dim);

      for(j = 0; j < dim; j++)
        fscanf(fpart, "%d ", &(t->interpart[mode][j]));

      fscanf(fpart, "\n#%d\n", &layersize);

      t->intrapart[mode] = (int *)myMalloc(sizeof(int)*dim);
		
      for(j = 0; j < dim; j++)
        fscanf(fpart, "%d ", &(t->intrapart[mode][j]));
    }
	
  fclose(fpart);
}


void read_ckbd_tensor_nonzeros(char tensorfile[], struct tensor *t)
{
  int i, offset,  nmodes, nnz, *inds, val, total, *buf, *tmp;
  double *vals;
  char line[1024], *str;
  MPI_File fh;
  MPI_Status status;

  long long int of;
  
  nmodes = t->nmodes;
  offset = t->mype*3*sizeof(int);
  buf = (int *)malloc((nmodes+1)*sizeof(int));

  MPI_File_open(MPI_COMM_WORLD, tensorfile, MPI_MODE_RDONLY, MPI_INFO_NULL, &fh);
  MPI_File_read_at(fh, offset, buf, 3, MPI_INT, &status);
  nnz = buf[1];
  of = buf[2]*(nmodes+1)*sizeof(int)+t->npes*3*sizeof(int);
  t->inds = (int *)myMalloc(nmodes*nnz*sizeof(int));
  t->vals = (double *)myMalloc(nnz*sizeof(double));
  t->nnz = nnz;

  tmp = (int *)myMalloc((nmodes+1)*nnz*sizeof(int));

  MPI_Allreduce(&nnz, &t->gnnz, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

  inds = t->inds;
  vals = t->vals;

  MPI_File_read_at(fh, of, tmp, (nmodes+1)*nnz, MPI_INT, &status);
  MPI_File_close(&fh);

  for(i = 0; i < nnz; i++)
    {
      memcpy(&inds[i*nmodes], &tmp[i*(nmodes+1)], sizeof(int)*nmodes);
      vals[i] = (double) tmp[i*(nmodes+1)+nmodes];
     }
  free(buf);

}

void read_ckbd_tensor_nonzeros_endian(char tensorfile[], struct tensor *t)
{
  int i, j, nmodes, nnz, *inds, val, *buff, total;
  double *vals;
  char line[1024], *str;
  FILE *ftensor;
	
  nmodes = t->nmodes;
  buff = (int *)malloc((nmodes+2)*sizeof(int));

  ftensor = fopen(tensorfile, "rb");    
  fread(buff, sizeof(int), nmodes+2, ftensor);
  for(j = 0; j < nmodes+2; j++)
    buff[j] = convert(buff[j]);
  
  t->gnnz = buff[nmodes+1];
	
  fread(&nnz, sizeof(int), 1, ftensor);
  nnz = convert(nnz);

  t->inds = (int *)myMalloc(nmodes*nnz*sizeof(int));
  t->vals = (double *)myMalloc(nnz*sizeof(double));
  t->nnz = nnz;
	
  inds = t->inds;
  vals = t->vals;

  int b = 0;
  for(i = 0; i < nnz; i++)
    {
      fread(&inds[b], sizeof(int), nmodes, ftensor);
      for(j = 0; j < nmodes; j++)
        inds[b+j] = convert(inds[b+j]);
      b += nmodes;

      fread(&val, sizeof(int), 1, ftensor);
      val = convert(val);
      vals[i] = (double) val;
      
      if(vals[i] == 0)
        vals[i] = 1.1;
    }
  fclose(ftensor);
  free(buff);
}


void read_ckbd_tensor(char tensorfile[], char partfile[], struct tensor *t, int endian)
{
  /* if(endian) */
  /*   read_dimensions_bin_endian(tensorfile, t); */
  /* else */
  /*   read_dimensions_bin(tensorfile, t); */
	
  read_ckbd_partition(partfile, t);
	
  split_communicators(t);

  if(endian)
    read_ckbd_tensor_nonzeros_endian(tensorfile, t);
  else
    read_ckbd_tensor_nonzeros(tensorfile, t);

}

void read_fg_partition(char partfile[], struct tensor *t)
{
  int i, j, nmodes, npes, dim;
  FILE *fpart;
  char tname[1024];


  fpart = fopen(partfile, "r");
  fscanf(fpart, "##%d %d", &nmodes, &npes);
  t->nmodes = nmodes;

  if(npes != t->npes)
    {
      if(t->mype == 0)
        printf("The partition file is for %d processors!!\n", npes);
      MPI_Finalize();
      exit(1);
    }

  t->gdims = (int *)malloc(nmodes*sizeof(int));

  t->interpart = (int **)myMalloc(sizeof(int *)*nmodes);
  for(i = 0; i < nmodes; i++)
    {
      fscanf(fpart, "\n#%d", &dim);
      t->gdims[i] = dim; 
		
      t->interpart[i] = (int *)myMalloc(sizeof(int)*dim);

      for(j = 0; j < dim; j++)
        fscanf(fpart, "\n%d", &(t->interpart[i][j]));
    }
	
  fclose(fpart);
}


void read_fg_tensor(char tensorfile[], char partfile[], struct tensor *t, int endian)
{
  read_fg_partition(partfile, t);

  if(endian)
    read_ckbd_tensor_nonzeros_endian(tensorfile, t);
  else
    read_ckbd_tensor_nonzeros(tensorfile, t);

}


int main(int argc, char *argv[])
{
  //int mype, npes, niters, i, endian;
  int mype, npes, i, endian;
  double readtime, setuptime, cptime,cptime2, cptimewb, totaltime, ltime[3], gtime[3];
	
  char tensorfile[1024], partfile[1024], meshstr[1024]; 
  struct tensor *t;
  struct stat *st;

  MPI_Init(&argc, &argv);

  totaltime = (double )get_wc_time();

  MPI_Comm_rank(MPI_COMM_WORLD, &mype);
  MPI_Comm_size(MPI_COMM_WORLD, &npes);
	
  t = (struct tensor *)myMalloc(sizeof(struct tensor));
  t->mype = mype;
  t->npes = npes;

  init_tensor(t);
  //init_param(argc, argv, tensorfile, partfile, meshstr, t, &niters, &endian);
  init_param(argc, argv, tensorfile, partfile, meshstr, t, NULL, &endian);

  if(t->ckbd == 1) //medium-grain
    {
      //read tensor
      readtime = (double) get_wc_time();
      read_ckbd_tensor(tensorfile, partfile, t, endian);
	
      MPI_Barrier(MPI_COMM_WORLD);
      readtime = ((double) get_wc_time() - readtime)/1000000;
		
      st = (struct stat *)myMalloc(sizeof(struct stat));
      init_stat(st, t->nmodes);

      //setup the environment
      setuptime = (double) get_wc_time();
      setup_ckbd_communication(t, st);

    }
  else if(t->ckbd == 2) //fine-grain
    {
      //read tensor
      readtime = (double) get_wc_time();
      read_fg_tensor(tensorfile, partfile, t, endian);

      MPI_Barrier(MPI_COMM_WORLD);
      readtime = ((double) get_wc_time() - readtime)/1000000;
		
      st = (struct stat *)myMalloc(sizeof(struct stat));
      init_stat(st, t->nmodes);

      //setup the environment
      setuptime = (double) get_wc_time();
      setup_fg_communication(t, st);
    }
  else
    {
      //read tensor
      readtime = (double) get_wc_time();
      read_cyclic_tensor(tensorfile, partfile, meshstr, t);

      MPI_Barrier(MPI_COMM_WORLD);
      readtime = ((double) get_wc_time() - readtime)/1000000;

      st = (struct stat *)myMalloc(sizeof(struct stat));
      init_stat(st, t->nmodes);
		
      //setup the environment
      setuptime = (double) get_wc_time();
      setup_cyclic_communication(t, st);
    }

  struct fibertensor *ft = NULL;
  if(t->fiber)
    {
      ft = (struct fibertensor *)myMalloc(sizeof(struct fibertensor));
      init_fibertensor(ft);
      get_fibertensor(t, ft);
    }

  init_matrices(t);

  MPI_Barrier(MPI_COMM_WORLD);
  setuptime = ((double) get_wc_time() - setuptime)/1000000;

  /* if(t->ckbd == 2) */
  /*   cp_als_barrier_fg(t, ft, niters); */
  /* else */
  /*   cp_als_barrier(t, ft, niters); */

int iter=-1;
  // cp factorization
  if(t->ckbd == 2){
    double cptimeW,cptime2W;
	cp_als_fg(t, ft, WARMUP, &cptimeW,&cptime2W);
	double wgcptime;
	MPI_Reduce(&cptimeW, &wgcptime, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
	int liter1=(int)(2.0/wgcptime)+1;
	if (liter1<10)
		liter1=10;
	int iterx=-1;
	MPI_Allreduce(&liter1, &iterx, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    cp_als_fg(t, ft, iterx, &cptimeW,&cptime2W);
	
	
	
	
	
	
	MPI_Reduce(&cptimeW, &wgcptime, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
	int liter=(int)(TOTTIME/wgcptime)+1;
	if (liter<10)
		liter=10;
	MPI_Allreduce(&liter, &iter, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    cp_als_fg(t, ft, iter, &cptime,&cptime2);
	
	
	
	
	
	
	
	
	
	
	
  }
  else
    cp_als(t, ft, -1, &cptime);
    //cp_als(t, ft, niters, &cptime);

  double *comm1time, *comm2time, *mttkrptime;
  comm1time = (double *)calloc(t->nmodes, sizeof(double));
  comm2time = (double *)calloc(t->nmodes, sizeof(double));
  mttkrptime = (double *)calloc(t->nmodes, sizeof(double));

  int *cnt_st = (int *)calloc(t->nmodes*4, sizeof(int));
  
/*  // for comm comp timing purposes
  if(t->ckbd == 2)
	  cp_als_stat_fg(t, ft, niters, mttkrptime, comm1time, comm2time, cnt_st);
  else
	  cp_als_stat(t, ft, niters, mttkrptime, comm1time, comm2time, cnt_st);
*/
  print_stat(tensorfile, t->meshdims, st, t->nmodes, t->mype, t->npes, cptime,cptime2, mttkrptime, comm1time, comm2time, t->gnnz, t->ckbd, cnt_st,iter);

  free(comm1time);
  free(comm2time);
  free(mttkrptime);
  free(cnt_st);
  if(t->fiber)
    free_fibertensor(ft, t->nmodes);
  free_stat(st);

  MPI_Barrier(MPI_COMM_WORLD);
  totaltime = ((double )get_wc_time() - totaltime)/1000000;

  if(t->mype == 0)
    printf("%.2f\t%.2f\t%.2f\t...\n", readtime, setuptime, totaltime);
		

  free_tensor(t);
  MPI_Finalize();

  return 0;
}

