/**
 * @file
 * This file is part of SWE.
 *
 * @author Michael Bader (bader AT in.tum.de, http://www5.in.tum.de/wiki/index.php/Univ.-Prof._Dr._Michael_Bader)
 * @author Alexander Breuer (breuera AT in.tum.de, http://www5.in.tum.de/wiki/index.php/Dipl.-Math._Alexander_Breuer)
 * @author Sebastian Rettenberger (rettenbs AT in.tum.de, http://www5.in.tum.de/wiki/index.php/Sebastian_Rettenberger,_M.Sc.)
 *
 * @section LICENSE
 *
 * SWE is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * SWE is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with SWE.  If not, see <http://www.gnu.org/licenses/>.
 *
 *
 * @section DESCRIPTION
 *
 * Setting of SWE, which uses a wave propagation solver and an artificial or ASAGI scenario on multiple blocks.
 */

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <GASPI.h>
#include <GASPI_Ext.h>
#include <GPI2_Stats.h>
#include <string>
#include <vector>

#include "blocks/SWE_Block.hh"
#include "blocks/SWE_WavePropagationBlock.hh"
#include "writer/Writer.hh"

#ifdef ASAGI
#include "scenarios/SWE_AsagiScenario.hh"
#else
#include "scenarios/SWE_simple_scenarios.hh"
#endif

#include "tools/args.hh"
#include "tools/help.hh"
#include "tools/Logger.hh"
#include "tools/ProgressBar.hh"

void success_or_exit (const char *file, const int line, const gaspi_return_t ec)
{
  if (ec != GASPI_SUCCESS)
  {
    tools::Logger::logger.cout() << "Assertion failed in " << file 
                               << "[" << line << "]: Return " << ec 
                               << ": " <<  gaspi_error_str (ec) << std::endl;
    exit(EXIT_FAILURE);
  }
}

#define ASSERT(ec)  success_or_exit (__FILE__, __LINE__, ec)

typedef struct flow_info_t
{
  int l_nleftOutflowOffset;
  int l_nleftInflowOffset;
  int l_nrightOutflowOffset;
  int l_nrightInflowOffset;
  gaspi_offset_t *l_leftOutflowOffset;
  gaspi_offset_t *l_leftInflowOffset;
  gaspi_offset_t *l_rightOutflowOffset;
  gaspi_offset_t *l_rightInflowOffset;
  int l_nbottomOutflowOffset;
  int l_nbottomInflowOffset;
  int l_ntopOutflowOffset;
  int l_ntopInflowOffset;
  gaspi_offset_t *l_bottomOutflowOffset;
  gaspi_offset_t *l_bottomInflowOffset;
  gaspi_offset_t *l_topOutflowOffset;
  gaspi_offset_t *l_topInflowOffset;
  int l_leftNeighborRank;
  int l_rightNeighborRank;
  int l_bottomNeighborRank;
  int l_topNeighborRank;
  gaspi_segment_id_t *segment_id_lr;
  gaspi_size_t *size_lr;
  gaspi_segment_id_t *segment_id_bt;
  gaspi_size_t *size_bt;
} flow_info_t;

class GPI_SWE
{
  public:
  int computeNumberOfBlockRows (int i_numberOfProcesses);
  void exchangeLeftRightGhostLayers ();
  void exchangeBottomTopGhostLayers ();
  gaspi_return_t recalculateRedistribute (int checkpt);
  gaspi_offset_t *calculateOffsets (Float2D& grid, BoundaryEdge edge, BoundaryType type);
  gaspi_return_t redistributeData (int l_oldRows, int l_oldCols);
  void startComputation ();
  std::ofstream *createFile (std::string type, int count);
  void collectStepbyStepStats (int spawned_procs);
  GPI_SWE (int argc, char** argv);
  ~GPI_SWE ();

  private:
  gaspi_rank_t l_gpiRank;
  gaspi_rank_t l_numberOfProcesses;
  flow_info_t l_flow;
  float l_t;
  unsigned int l_iterations;
  int l_startc;
  gaspi_config_t config;
  tools::Args args;
  //! total number of grid cell in x- and y-direction.
  int l_nX, l_nY;
  //! l_baseName of the plots.
  std::string l_baseName;
  //! number of SWE_Blocks in x- and y-direction.
  int l_blocksX, l_blocksY;
  //! local position of each MPI process in x- and y-direction.
  int l_blockPositionX, l_blockPositionY;
  SWE_BathymetryDamBreakScenario l_scenario;
  int l_numberOfCheckPoints;
  //! number of grid cells in x- and y-direction per process.
  int l_nXLocal, l_nYLocal;
  int l_nXNormal, l_nYNormal;
  //! size of a single cell in x- and y-direction
  float l_dX, l_dY;
  //! origin of the simulation domain in x- and y-direction
  float l_originX, l_originY;
  float *l_height, *l_discharge_hu, *l_discharge_hv;
  SWE_Block *l_waveBlock;
  int spawned;
  float l_endSimulation;
  float* l_checkPoints;
  std::string l_fileName;
  io::BoundarySize l_boundarySize;
  std::shared_ptr<io::Writer> l_writer;
  int argc;
  char** argv;
};

GPI_SWE::GPI_SWE (int l_argc, char** l_argv) : 
            l_t (0.0), l_iterations (0), l_startc (1), spawned (0),
            l_boundarySize ({{1, 1, 1, 1}}), argc (l_argc), argv (l_argv) {
  ASSERT(gaspi_config_get(&config));
  config.queue_size_max = 1024;
  config.rw_list_elem_max = 1024;
  ASSERT(gaspi_config_set(config));
  ASSERT(gaspi_proc_init(GASPI_BLOCK));
  ASSERT(gaspi_proc_rank(&l_gpiRank));
  ASSERT(gaspi_proc_num(&l_numberOfProcesses));
  std::cerr << "my rank " << l_gpiRank << " number of procs " << l_numberOfProcesses << std::endl;
  tools::Logger::logger.setProcessRank(l_gpiRank);
  //tools::Logger::logger.printWelcomeMessage();
  // set current wall clock time within the solver
  tools::Logger::logger.initWallClockTime(time(NULL));
  tools::Logger::logger.printNumberOfProcesses(l_numberOfProcesses);
  args.addOption("grid-size-x", 'x', "Number of cell in x direction");
  args.addOption("grid-size-y", 'y', "Number of cell in y direction");
  args.addOption("output-basepath", 'o', "Output base file name");
  args.addOption("output-steps-count", 'c', "Number of output time steps");
  tools::Args::Result ret = args.parse(argc, argv, l_gpiRank == 0);
  switch (ret)
  {
    case tools::Args::Error:
      exit(EXIT_FAILURE);
    case tools::Args::Help:
      ASSERT(gaspi_proc_term(GASPI_BLOCK));
      return;
    default:
        break;
  }
  l_nX = args.getArgument<int>("grid-size-x");
  l_nY = args.getArgument<int>("grid-size-y");
  l_baseName = args.getArgument<std::string>("output-basepath");
  l_numberOfCheckPoints = args.getArgument<int>("output-steps-count");
  l_blocksY = l_numberOfProcesses;
  l_blocksX = l_numberOfProcesses/l_blocksY;
  tools::Logger::logger.printNumberOfCells(l_nX, l_nY);
  tools::Logger::logger.printNumberOfBlocks(l_blocksX, l_blocksY);
  // determine local block coordinates of each SWE_Block
  l_blockPositionX = l_gpiRank / l_blocksY;
  l_blockPositionY = l_gpiRank % l_blocksY;
  // compute local number of cells for each SWE_Block
  l_nXLocal = (l_blockPositionX < l_blocksX-1) ? l_nX/l_blocksX : \
                                  l_nX - (l_blocksX-1)*(l_nX/l_blocksX);
  l_nYLocal = (l_blockPositionY < l_blocksY-1) ? l_nY/l_blocksY : \
                                  l_nY - (l_blocksY-1)*(l_nY/l_blocksY);
  l_nXNormal = l_nX/l_blocksX;
  l_nYNormal = l_nY/l_blocksY;

  // compute the size of a single cell
  l_dX = (l_scenario.getBoundaryPos(BND_RIGHT) - l_scenario.getBoundaryPos(BND_LEFT) )/l_nX;
  l_dY = (l_scenario.getBoundaryPos(BND_TOP) - l_scenario.getBoundaryPos(BND_BOTTOM) )/l_nY;

  // print information about the cell size and local number of cells
  tools::Logger::logger.printCellSize(l_dX, l_dY);
  tools::Logger::logger.printNumberOfCellsPerProcess(l_nXLocal, l_nYLocal);
  // get the origin from the scenario
  l_originX = l_scenario.getBoundaryPos(BND_LEFT) + l_blockPositionX*l_nXNormal*l_dX;;
  l_originY = l_scenario.getBoundaryPos(BND_BOTTOM) + l_blockPositionY*l_nYNormal*l_dY;
  int rows = l_nXLocal+2;
  int cols = l_nYLocal+2;
  l_height = new float[rows*cols];
  l_discharge_hu = new float[rows*cols];
  l_discharge_hv = new float[rows*cols];
  Float2D l_h(rows, cols, l_height);
  Float2D l_hu(rows, cols, l_discharge_hu);
  Float2D l_hv(rows, cols, l_discharge_hv);
  l_waveBlock = SWE_Block::getBlockInstance(l_nXLocal,
                                            l_nYLocal, 
                                            l_dX, 
                                            l_dY,
                                            l_h,
                                            l_hu,
                                            l_hv);
  ASSERT(gaspi_segment_use(0, (gaspi_pointer_t) l_height,
                           (rows) * (cols) * sizeof(float),
                           GASPI_GROUP_ALL, GASPI_BLOCK, 0));
  ASSERT(gaspi_segment_use(1, (gaspi_pointer_t) l_discharge_hu,
                           (rows) * (cols) * sizeof(float),
                           GASPI_GROUP_ALL, GASPI_BLOCK, 0));
  ASSERT(gaspi_segment_use(2, (gaspi_pointer_t) l_discharge_hv,
                           (rows) * (cols) * sizeof(float),
                           GASPI_GROUP_ALL, GASPI_BLOCK, 0));
  l_waveBlock->initScenario(l_originX, l_originY, l_scenario, true);
  ASSERT(gaspi_barrier(GASPI_GROUP_ALL, GASPI_BLOCK));
  ASSERT(gaspi_spawned(&spawned));
  if(spawned)
  {
    gaspi_notification_id_t fid;
    gaspi_notification_t val;
    for(int j = 0; j < 3; j++)
    {
      ASSERT(gaspi_notify_waitsome(j, 1, 1, &fid, GASPI_BLOCK));
      val = 0;
      ASSERT (gaspi_notify_reset(j, fid, &val));
    }
    ASSERT(gaspi_barrier(GASPI_GROUP_ALL, GASPI_BLOCK));
  }
  l_endSimulation = l_scenario.endSimulation();
  l_checkPoints = new float[l_numberOfCheckPoints+1];
  for(int cp = 0; cp <= l_numberOfCheckPoints; cp++) {
     l_checkPoints[cp] = cp*(l_endSimulation/l_numberOfCheckPoints);
  }
  if (l_blockPositionX == 0)
    l_waveBlock->setBoundaryType(BND_LEFT, OUTFLOW);
  if (l_blockPositionX == l_blocksX-1)
    l_waveBlock->setBoundaryType(BND_RIGHT, OUTFLOW);
  if (l_blockPositionY == 0)
    l_waveBlock->setBoundaryType(BND_BOTTOM, OUTFLOW);
  if (l_blockPositionY == l_blocksY-1)
    l_waveBlock->setBoundaryType(BND_TOP, OUTFLOW);
  l_flow.l_nleftOutflowOffset = l_waveBlock->getWaterHeight().getRows();
  l_flow.l_nleftInflowOffset = l_flow.l_nleftOutflowOffset;
  l_flow.l_nrightOutflowOffset = l_flow.l_nleftOutflowOffset;
  l_flow.l_nrightInflowOffset = l_flow.l_nleftOutflowOffset;
  l_flow.l_leftOutflowOffset = calculateOffsets(l_h, BND_LEFT, OUTFLOW);
  l_flow.l_leftInflowOffset = calculateOffsets(l_h, BND_LEFT, INFLOW);
  l_flow.l_rightOutflowOffset = calculateOffsets(l_h, BND_RIGHT, OUTFLOW);
  l_flow.l_rightInflowOffset = calculateOffsets(l_h, BND_RIGHT, INFLOW);

  l_flow.l_nbottomOutflowOffset = l_waveBlock->getWaterHeight().getCols();
  l_flow.l_nbottomInflowOffset = l_flow.l_nbottomOutflowOffset;
  l_flow.l_ntopOutflowOffset = l_flow.l_nbottomOutflowOffset;
  l_flow.l_ntopInflowOffset = l_flow.l_nbottomOutflowOffset;
  l_flow.l_bottomOutflowOffset = calculateOffsets(l_h, BND_BOTTOM, OUTFLOW);
  l_flow.l_bottomInflowOffset = calculateOffsets(l_h, BND_BOTTOM, INFLOW);
  l_flow.l_topOutflowOffset = calculateOffsets(l_h, BND_TOP, OUTFLOW);
  l_flow.l_topInflowOffset = calculateOffsets(l_h, BND_TOP, INFLOW);

  l_flow.l_leftNeighborRank   = (l_blockPositionX > 0) ? l_gpiRank-l_blocksY : -1;
  l_flow.l_rightNeighborRank  = (l_blockPositionX < l_blocksX-1) ? l_gpiRank+l_blocksY : -1;
  l_flow.l_bottomNeighborRank = (l_blockPositionY > 0) ? l_gpiRank-1 : -1;
  l_flow.l_topNeighborRank    = (l_blockPositionY < l_blocksY-1) ? l_gpiRank+1 : -1;

  l_flow.segment_id_lr = (gaspi_segment_id_t *) calloc (l_flow.l_nleftOutflowOffset,
                                                        sizeof(gaspi_segment_id_t));
  l_flow.size_lr = (gaspi_size_t *) calloc (l_flow.l_nleftOutflowOffset, 
                                            sizeof(gaspi_size_t));
  exchangeLeftRightGhostLayers();
  l_flow.segment_id_bt = (gaspi_segment_id_t *) calloc (l_flow.l_nbottomOutflowOffset,
                                                        sizeof(gaspi_segment_id_t));
  l_flow.size_bt = (gaspi_size_t *) calloc (l_flow.l_nbottomOutflowOffset,
                                            sizeof(gaspi_size_t));
  exchangeBottomTopGhostLayers();
  ASSERT(gaspi_barrier(GASPI_GROUP_ALL, GASPI_BLOCK));
  tools::Logger::logger.printOutputTime(0);
  l_fileName = generateBaseFileName(l_baseName,l_blockPositionX,l_blockPositionY);
  if(spawned)
  {
    gaspi_allreduce(&l_t, &l_t, 1,
                    GASPI_OP_MAX, GASPI_TYPE_FLOAT,
                    GASPI_GROUP_ALL, GASPI_BLOCK);
    gaspi_allreduce(&l_iterations, &l_iterations, 1,
                    GASPI_OP_MAX, GASPI_TYPE_UINT,
                    GASPI_GROUP_ALL, GASPI_BLOCK);
    gaspi_allreduce(&l_startc, &l_startc, 1,
                    GASPI_OP_MAX, GASPI_TYPE_INT,
                    GASPI_GROUP_ALL, GASPI_BLOCK);
    collectStepbyStepStats (l_numberOfProcesses/2);
    l_writer = io::Writer::createWriterInstance(
          l_fileName,
          l_waveBlock->getBathymetry(),
          l_boundarySize,
          l_nXLocal, l_nYLocal,
          l_dX, l_dY,
          l_blockPositionX*l_nXLocal, l_blockPositionY*l_nYLocal,
          l_originX, l_originY,
          0, l_startc-1);
  }
  else
  {
    l_writer = io::Writer::createWriterInstance(
          l_fileName,
          l_waveBlock->getBathymetry(),
          l_boundarySize,
          l_nXLocal, l_nYLocal,
          l_dX, l_dY,
          l_blockPositionX*l_nXLocal, l_blockPositionY*l_nYLocal,
          l_originX, l_originY,
          0);
    /*l_writer->writeTimeStep( l_waveBlock->getWaterHeight(),
                            l_waveBlock->getDischarge_hu(),
                            l_waveBlock->getDischarge_hv(),
                            (float) l_t);*/
    tools::Logger::logger.printStartMessage();
  }
}


int GPI_SWE::computeNumberOfBlockRows(int i_numberOfProcesses) {
  return i_numberOfProcesses;
}

std::ofstream *GPI_SWE::createFile (std::string typeFile, int count)
{
  std::ostringstream l_file;
  l_file << l_baseName << "_" << typeFile << "_" << count << ".txt";
  std::string tempName = l_file.str();
  std::ofstream *newFile = new std::ofstream(tempName.c_str());
  assert(newFile->good());
  return newFile;
}

void GPI_SWE::collectStepbyStepStats (int spawned_procs)
{
  /*First collect the time required to spawn new procs*/
  float spawn_time = 0;
  if (l_gpiRank == 0)
  {
    spawn_time = GPI2_STATS_GET_TIMER (GASPI_SPAWN_TIMER)/1000;
    std::ofstream *spawnFile = createFile ("spawn_time", spawned_procs);
    *spawnFile << spawn_time << std::endl;
    GPI2_STATS_RESET_TIMER (GASPI_SPAWN_TIMER);
  }
  /*Collect the time required for all procs to be notified of the
  completion of spawned procs. This includes the time required to
  spawn the procs.*/
  float init_resource_change = GPI2_STATS_GET_TIMER (GASPI_INIT_RESOURCE_CHANGE_TIMER)/1000;
  /*Average it out over the procs involved*/
  float total_resource_change = 0;
  gaspi_allreduce(&init_resource_change, &total_resource_change, 1,
                  GASPI_OP_SUM, GASPI_TYPE_FLOAT, GASPI_GROUP_ALL, GASPI_BLOCK);
  if (l_gpiRank == 0)
  {
    total_resource_change /= spawned_procs;
    std::ofstream *spawnFile = createFile ("init_resource_change_time", spawned_procs);
    *spawnFile << total_resource_change << std::endl;
  }
  GPI2_STATS_RESET_TIMER (GASPI_INIT_RESOURCE_CHANGE_TIMER);
  /*Collect the time required to construct a PMIx Group.*/
  float group_construct = GPI2_STATS_GET_TIMER (GASPI_GROUP_CONSTRUCT_TIMER)/1000;
  /*Average it out over the procs involved*/
  float total_group_construct = 0;
  gaspi_allreduce(&group_construct, &total_group_construct, 1,
                  GASPI_OP_SUM, GASPI_TYPE_FLOAT, GASPI_GROUP_ALL, GASPI_BLOCK);
  if (l_gpiRank == 0)
  {
    total_group_construct /= l_numberOfProcesses;
    std::ofstream *spawnFile = createFile ("group_construct", spawned_procs);
    *spawnFile << total_group_construct << std::endl;
  }
  GPI2_STATS_RESET_TIMER (GASPI_GROUP_CONSTRUCT_TIMER);
  /*Collect the time required to connect PMIx namespaces.*/
  float connect_nspace = GPI2_STATS_GET_TIMER (GASPI_PMIX_CONNECT_TIMER)/1000;
  /*Average it out over the procs involved*/
  float total_connect_nspace = 0;
  gaspi_allreduce(&connect_nspace, &total_connect_nspace, 1,
                  GASPI_OP_SUM, GASPI_TYPE_FLOAT, GASPI_GROUP_ALL, GASPI_BLOCK);
  if (l_gpiRank == 0)
  {
    total_connect_nspace /= l_numberOfProcesses;
    std::ofstream *spawnFile = createFile ("connect_nspace", spawned_procs);
    *spawnFile << total_connect_nspace << std::endl;
  }
  GPI2_STATS_RESET_TIMER (GASPI_PMIX_CONNECT_TIMER);
  /*Collect the time required to init/reinit gaspi core.*/
  float init_core = GPI2_STATS_GET_TIMER (GASPI_CORE_CONTEXT_INIT_TIMER)/1000;
  /*Average it out over the procs involved*/
  float total_init_core = 0;
  gaspi_allreduce(&init_core, &total_init_core, 1,
                  GASPI_OP_SUM, GASPI_TYPE_FLOAT, GASPI_GROUP_ALL, GASPI_BLOCK);
  if (l_gpiRank == 0)
  {
    total_init_core /= l_numberOfProcesses;
    std::ofstream *spawnFile = createFile ("init_reinit_core", spawned_procs);
    *spawnFile << total_init_core << std::endl;
  }
  GPI2_STATS_RESET_TIMER (GASPI_CORE_CONTEXT_INIT_TIMER);
  /*Collect the time required to create gaspi groups and dynamic connections.*/
  float gaspi_grp_conn = GPI2_STATS_GET_TIMER (GASPI_CONNECTION_GROUP_CREATION_TIMER)/1000;
  /*Average it out over the procs involved*/
  float total_gaspi_grp_conn = 0;
  gaspi_allreduce(&gaspi_grp_conn, &total_gaspi_grp_conn, 1,
                  GASPI_OP_SUM, GASPI_TYPE_FLOAT, GASPI_GROUP_ALL, GASPI_BLOCK);
  if (l_gpiRank == 0)
  {
    total_gaspi_grp_conn /= l_numberOfProcesses;
    std::ofstream *spawnFile = createFile ("gaspi_group_create_conn", spawned_procs);
    *spawnFile << total_gaspi_grp_conn << std::endl;
  }
  GPI2_STATS_RESET_TIMER (GASPI_CONNECTION_GROUP_CREATION_TIMER);
  /*Collect the total resource change time.*/
  float total_time = GPI2_STATS_GET_TIMER (GASPI_TOTAL_RESOURCE_CHANGE_TIME)/1000;
  /*Average it out over the procs involved*/
  float sum_total_time = 0;
  gaspi_allreduce(&total_time, &sum_total_time, 1,
                  GASPI_OP_SUM, GASPI_TYPE_FLOAT, GASPI_GROUP_ALL, GASPI_BLOCK);
  if (l_gpiRank == 0)
  {
    sum_total_time /= l_numberOfProcesses;
    std::ofstream *spawnFile = createFile ("reconf", spawned_procs);
    *spawnFile << sum_total_time << std::endl;
  }
  GPI2_STATS_RESET_TIMER (GASPI_TOTAL_RESOURCE_CHANGE_TIME);
  GPI2_STATS_RESET_ALL_TIMERS;
  GPI2_STATS_RESET_ALL_COUNTERS;
}

void GPI_SWE::startComputation ()
{
  tools::Logger::logger.initWallClockTime(time(NULL));
  // loop over checkpoints
  double l_lastIterTime = clock();
  double l_lastCheckTime = 0;
  std::ofstream *iterFile = NULL;
  std::ofstream *checkFile = NULL;
  for(int c=l_startc; c<=l_numberOfCheckPoints; c++) {
    if (l_gpiRank == 0)
    {
      iterFile = createFile ("iterations", c); 
      checkFile = createFile ("checkpoint", c);
    }  
    while( l_t < l_checkPoints[c] ) {
      tools::Logger::logger.resetClockToCurrentTime("Cpu");
      exchangeLeftRightGhostLayers ();
      exchangeBottomTopGhostLayers ();
      ASSERT(gaspi_barrier(GASPI_GROUP_ALL, GASPI_BLOCK));
      l_waveBlock->setGhostLayer();
      l_waveBlock->computeNumericalFluxes();
      float l_maxTimeStepWidth = l_waveBlock->getMaxTimestep();
      float l_maxTimeStepWidthGlobal;
      gaspi_allreduce(&l_maxTimeStepWidth, &l_maxTimeStepWidthGlobal, 1,
                      GASPI_OP_MIN, GASPI_TYPE_FLOAT, GASPI_GROUP_ALL, GASPI_BLOCK);
      l_waveBlock->updateUnknowns(l_maxTimeStepWidthGlobal);
      tools::Logger::logger.updateTime("Cpu");
      l_t += l_maxTimeStepWidthGlobal;
      l_iterations++;
      /*Compute average time spent per iteration for overall computation.*/
      double l_iterTime = tools::Logger::logger.getTime("Cpu") - l_lastIterTime;
      l_lastIterTime = tools::Logger::logger.getTime("Cpu");
      double l_sumIterTime = 0;
      gaspi_allreduce(&l_iterTime, &l_sumIterTime, 1,
                      GASPI_OP_SUM, GASPI_TYPE_DOUBLE, GASPI_GROUP_ALL, GASPI_BLOCK);
      if (l_gpiRank == 0)
      {
        l_sumIterTime /= l_numberOfProcesses;
        *iterFile << l_sumIterTime << std::endl;
      }
      // print the current simulation time
      tools::Logger::logger.printSimulationTime(l_t);
    }
    /*Compute total simulation time spent per checkpoint.*/
    double l_sumCheckTime = 0, l_checkTime;
    l_checkTime = l_lastIterTime - l_lastCheckTime;
    l_lastCheckTime = l_lastIterTime;
    gaspi_allreduce(&l_checkTime, &l_sumCheckTime, 1,
                    GASPI_OP_SUM, GASPI_TYPE_DOUBLE, GASPI_GROUP_ALL, GASPI_BLOCK);
    if (l_gpiRank == 0)
    {
      l_sumCheckTime /= l_numberOfProcesses;
      *checkFile << l_sumCheckTime << std::endl;
    }

    // write output
    /*l_writer->writeTimeStep (l_waveBlock->getWaterHeight(),
                             l_waveBlock->getDischarge_hu(),
                             l_waveBlock->getDischarge_hv(),
                             l_t);*/
    /*We want to now add more resources to the computation.
    We request l_numberOfProcesses of additional processes.*/
    //if(c+1 == 5 || c+1 == 9 || c+1 == 13)
    if (c+1 <= l_numberOfCheckPoints)
    {
      ASSERT(gaspi_proc_alloc_request(l_numberOfProcesses,
                                      &l_gpiRank, 
                                      &l_numberOfProcesses,
                                      argc, argv));
      ASSERT(recalculateRedistribute (c));
      gaspi_allreduce(&l_t, &l_t, 1,
                      GASPI_OP_MAX, GASPI_TYPE_FLOAT,
                      GASPI_GROUP_ALL, GASPI_BLOCK);
      gaspi_allreduce(&l_iterations, &l_iterations, 1,
                      GASPI_OP_MAX, GASPI_TYPE_UINT,
                      GASPI_GROUP_ALL, GASPI_BLOCK);
      int send_startc = c+1;
      gaspi_allreduce(&send_startc, &l_startc, 1,
                      GASPI_OP_MAX, GASPI_TYPE_INT,
                      GASPI_GROUP_ALL, GASPI_BLOCK);
      collectStepbyStepStats (l_numberOfProcesses/2);
    }
  }
}

GPI_SWE::~GPI_SWE ()
{
  delete l_waveBlock;
  tools::Logger::logger.printFinishMessage();
  ASSERT(gaspi_barrier(GASPI_GROUP_ALL, GASPI_BLOCK));
  ASSERT(gaspi_proc_term(GASPI_BLOCK));
}

/**
 * Main program for the simulation on a single SWE_WavePropagationBlock or SWE_WaveAccumulationBlock.
 */
int main (int argc, char** argv) {
  GPI_SWE l_gpiSwe (argc, argv);
  l_gpiSwe.startComputation ();
  return 0;
}

gaspi_return_t GPI_SWE::redistributeData(int l_oldRows, int l_oldCols)
{
  /*The data before resource addition is present in the above arguments.
  Here, we assume that old and new process ranks are the same.*/
  /*first we pin the blocks to segments*/
  ASSERT(gaspi_segment_bind(3, (gaspi_pointer_t) l_height,
                           (l_oldRows+2) * (l_oldCols+2) * sizeof(float),
                            0));
  ASSERT(gaspi_segment_bind(4, (gaspi_pointer_t) l_discharge_hu,
                           (l_oldRows+2) * (l_oldCols+2) * sizeof(float),
                            0));
  ASSERT(gaspi_segment_bind(5, (gaspi_pointer_t) l_discharge_hv,
                           (l_oldRows+2) * (l_oldCols+2) * sizeof(float),
                            0));
  /*first half goes to the same 2*rank, second half goes to 2*rank+1.*/
  int l_numCols = l_oldCols/2;
  gaspi_offset_t left_offsets[l_oldRows], right_offsets[l_oldRows];
  gaspi_offset_t left_remote_offsets[l_oldRows];
  gaspi_size_t sizes[l_oldRows];
  for(int i = 0; i < l_oldRows; i++)
  {
    left_offsets[i] = ((i+1)*(l_oldCols + 2) + 1) * sizeof(float);
    right_offsets[i] = ((i+1)*(l_oldCols + 2) + 1 + l_numCols) * sizeof(float);
    left_remote_offsets[i] = ((i+1)*(l_numCols + 2) + 1) * sizeof(float);
    sizes[i] = l_numCols * sizeof(float);
    /*tools::Logger::logger.cout() << "Writing from offset " << left_offsets[i] 
                               << " to offset " << left_remote_offsets[i] 
                               << " data of size " << sizes[i] << std::endl;*/
  }
  
  /*Now we write data to the respective ranks.*/
  gaspi_segment_id_t segment_src[l_oldRows];
  gaspi_segment_id_t segment_dest[l_oldRows];
  for(int i = 0; i < 3; i++)
  {
    for(int j = 0; j < l_oldRows; j++)
    {
      segment_src[j] = i+3;
      segment_dest[j] = i;
    }
    ASSERT(gaspi_write_list_notify(l_oldRows,
                                   segment_src,
                                   left_offsets,
                                   l_gpiRank * 2,
                                   segment_dest,
                                   left_remote_offsets,
                                   sizes,
                                   i,
                                   1,
                                   1,
                                   0,
                                   GASPI_BLOCK));
    ASSERT(gaspi_wait(0, GASPI_BLOCK));
    ASSERT(gaspi_write_list_notify(l_oldRows,
                                   segment_src,
                                   right_offsets,
                                   l_gpiRank * 2 + 1,
                                   segment_dest,
                                   left_remote_offsets,
                                   sizes,
                                   i,
                                   1,
                                   1,
                                   0,
                                   GASPI_BLOCK));
    ASSERT(gaspi_wait(0, GASPI_BLOCK));
  }
  /*Wait for our own data to arrive*/
  gaspi_notification_id_t fid;
  gaspi_notification_t val;
  for(int j = 0; j < 3; j++)
  {
    ASSERT(gaspi_notify_waitsome(j, 1, 1, &fid, GASPI_BLOCK));
    val = 0;
    ASSERT (gaspi_notify_reset(j, fid, &val));
  }
  ASSERT(gaspi_barrier(GASPI_GROUP_ALL, GASPI_BLOCK));
  tools::Logger::logger.cout() << "Received redistributed data " << std::endl;
  ASSERT(gaspi_segment_delete(3));
  ASSERT(gaspi_segment_delete(4));
  ASSERT(gaspi_segment_delete(5));
  return GASPI_SUCCESS;
}

gaspi_return_t GPI_SWE::recalculateRedistribute (int checkpoint)
{
  int l_oldRows = l_nXLocal;
  int l_oldCols = l_nYLocal;
  tools::Logger::logger.setProcessRank(l_gpiRank);
  l_blocksY = l_numberOfProcesses;
  l_blocksX = l_numberOfProcesses/l_blocksY;
  l_blockPositionX = l_gpiRank / l_blocksY;
  l_blockPositionY = l_gpiRank % l_blocksY;
  l_nXLocal = (l_blockPositionX < l_blocksX-1) ? l_nX/l_blocksX : l_nX - (l_blocksX-1)*(l_nX/l_blocksX);
  l_nYLocal = (l_blockPositionY < l_blocksY-1) ? l_nY/l_blocksY : l_nY - (l_blocksY-1)*(l_nY/l_blocksY);
  l_nXNormal = l_nX/l_blocksX;
  l_nYNormal = l_nY/l_blocksY;
  l_dX = (l_scenario.getBoundaryPos(BND_RIGHT) - l_scenario.getBoundaryPos(BND_LEFT) )/l_nX;
  l_dY = (l_scenario.getBoundaryPos(BND_TOP) - l_scenario.getBoundaryPos(BND_BOTTOM) )/l_nY;
  l_originX = l_scenario.getBoundaryPos(BND_LEFT) + l_blockPositionX*l_nXNormal*l_dX;;
  l_originY = l_scenario.getBoundaryPos(BND_BOTTOM) + l_blockPositionY*l_nYNormal*l_dY;
  int rows = l_nXLocal+2;
  int cols = l_nYLocal+2;
  float *new_height = new float[rows*cols];
  float *new_discharge_hu = new float[rows*cols];
  float *new_discharge_hv = new float[rows*cols];
  Float2D l_h(rows, cols, new_height);
  Float2D l_hu(rows, cols, new_discharge_hu);
  Float2D l_hv(rows, cols, new_discharge_hv);
  delete(l_waveBlock);
  l_waveBlock = SWE_Block::getBlockInstance(l_nXLocal,
                                            l_nYLocal, 
                                            l_dX, 
                                            l_dY,
                                            l_h,
                                            l_hu,
                                            l_hv);
  ASSERT(gaspi_segment_use(0, (gaspi_pointer_t) new_height,
                           (rows) * (cols) * sizeof(float),
                           GASPI_GROUP_ALL, GASPI_BLOCK, 0));
  ASSERT(gaspi_segment_use(1, (gaspi_pointer_t) new_discharge_hu,
                           (rows) * (cols) * sizeof(float),
                           GASPI_GROUP_ALL, GASPI_BLOCK, 0));
  ASSERT(gaspi_segment_use(2, (gaspi_pointer_t) new_discharge_hv,
                           (rows) * (cols) * sizeof(float),
                           GASPI_GROUP_ALL, GASPI_BLOCK, 0));
  
  l_waveBlock->initScenario(l_originX, l_originY, l_scenario, true);
  ASSERT(gaspi_barrier(GASPI_GROUP_ALL, GASPI_BLOCK));
  /*Now all procs have created new memory segments.
  We want to begin redistributing data*/
  ASSERT(redistributeData(l_oldRows, l_oldCols));

  /*After redistributing data, we can delete old pointers.*/
  free(l_height);
  free(l_discharge_hu);
  free(l_discharge_hv);
  l_height = new_height;
  l_discharge_hu = new_discharge_hu;
  l_discharge_hv = new_discharge_hv;
  
  if (l_blockPositionX == 0)
    l_waveBlock->setBoundaryType(BND_LEFT, OUTFLOW);
  if (l_blockPositionX == l_blocksX-1)
    l_waveBlock->setBoundaryType(BND_RIGHT, OUTFLOW);
  if (l_blockPositionY == 0)
    l_waveBlock->setBoundaryType(BND_BOTTOM, OUTFLOW);
  if (l_blockPositionY == l_blocksY-1)
    l_waveBlock->setBoundaryType(BND_TOP, OUTFLOW);
  
  l_flow.l_nleftOutflowOffset = l_waveBlock->getWaterHeight().getRows();
  l_flow.l_nleftInflowOffset = l_flow.l_nleftOutflowOffset;
  l_flow.l_nrightOutflowOffset = l_flow.l_nleftOutflowOffset;
  l_flow.l_nrightInflowOffset = l_flow.l_nleftOutflowOffset;
  free(l_flow.l_leftOutflowOffset);
  free(l_flow.l_leftInflowOffset);
  l_flow.l_leftOutflowOffset = calculateOffsets(l_h, BND_LEFT, OUTFLOW);
  l_flow.l_leftInflowOffset = calculateOffsets(l_h, BND_LEFT, INFLOW);
  free(l_flow.l_rightOutflowOffset);
  free(l_flow.l_rightInflowOffset);
  l_flow.l_rightOutflowOffset = calculateOffsets(l_h, BND_RIGHT, OUTFLOW);
  l_flow.l_rightInflowOffset = calculateOffsets(l_h, BND_RIGHT, INFLOW);

  l_flow.l_nbottomOutflowOffset = l_waveBlock->getWaterHeight().getCols();
  l_flow.l_nbottomInflowOffset = l_flow.l_nbottomOutflowOffset;
  l_flow.l_ntopOutflowOffset = l_flow.l_nbottomOutflowOffset;
  l_flow.l_ntopInflowOffset = l_flow.l_nbottomOutflowOffset;
  free(l_flow.l_bottomOutflowOffset);
  free(l_flow.l_bottomInflowOffset);
  l_flow.l_bottomOutflowOffset = calculateOffsets(l_h, BND_BOTTOM, OUTFLOW);
  l_flow.l_bottomInflowOffset = calculateOffsets(l_h, BND_BOTTOM, INFLOW);
  free(l_flow.l_topOutflowOffset);
  free(l_flow.l_topInflowOffset);
  l_flow.l_topOutflowOffset = calculateOffsets(l_h, BND_TOP, OUTFLOW);
  l_flow.l_topInflowOffset = calculateOffsets(l_h, BND_TOP, INFLOW);

  l_flow.l_leftNeighborRank   = (l_blockPositionX > 0) ? l_gpiRank-l_blocksY : -1;
  l_flow.l_rightNeighborRank  = (l_blockPositionX < l_blocksX-1) ? l_gpiRank+l_blocksY : -1;
  l_flow.l_bottomNeighborRank = (l_blockPositionY > 0) ? l_gpiRank-1 : -1;
  l_flow.l_topNeighborRank    = (l_blockPositionY < l_blocksY-1) ? l_gpiRank+1 : -1;

  // intially exchange ghost and copy layers
  free(l_flow.segment_id_lr);
  free(l_flow.size_lr);
  l_flow.segment_id_lr = (gaspi_segment_id_t *) calloc (l_flow.l_nleftOutflowOffset,
                                                        sizeof(gaspi_segment_id_t));
  l_flow.size_lr = (gaspi_size_t *) calloc (l_flow.l_nleftOutflowOffset,
                                            sizeof(gaspi_size_t));
  exchangeLeftRightGhostLayers ();
  free(l_flow.segment_id_bt);
  free(l_flow.size_bt);
  l_flow.segment_id_bt = (gaspi_segment_id_t *) calloc (l_flow.l_nbottomOutflowOffset,
                                                        sizeof(gaspi_segment_id_t));
  l_flow.size_bt = (gaspi_size_t *) calloc (l_flow.l_nbottomOutflowOffset,
                                            sizeof(gaspi_size_t));
  exchangeBottomTopGhostLayers ();
  ASSERT(gaspi_barrier(GASPI_GROUP_ALL, GASPI_BLOCK));
  l_fileName = generateBaseFileName(l_baseName,l_blockPositionX,l_blockPositionY);
  l_writer = io::Writer::createWriterInstance(
          l_fileName,
          l_waveBlock->getBathymetry(),
          l_boundarySize,
          l_nXLocal, l_nYLocal,
          l_dX, l_dY,
          l_blockPositionX*l_nXLocal, l_blockPositionY*l_nYLocal,
          l_originX, l_originY, 0,
          checkpoint);
  return GASPI_SUCCESS;
}

gaspi_offset_t *GPI_SWE::calculateOffsets(Float2D& grid, BoundaryEdge edge, BoundaryType type)
{
  int rows = grid.getRows(); /*nY*/
  int cols = grid.getCols(); /*nX*/
  float *base = grid.elemVector();
  gaspi_offset_t *offset = NULL;
  switch(edge)
  {
    case BND_LEFT:
      if (type == OUTFLOW)
      {
        float *first_ele = grid.getColProxy(1).elemVector();
        gaspi_offset_t start = first_ele - base;
        offset = (gaspi_offset_t *) calloc (rows, sizeof(gaspi_offset_t));
        for(int i = 0; i < rows; i++)
        {
          /*second row of grid*/
          offset[i] = (start + i) * sizeof(float); 
        }
      }
      else if (type == INFLOW)
      {
        float *first_ele = grid.getColProxy(0).elemVector();
        gaspi_offset_t start = first_ele - base;
        offset = (gaspi_offset_t *) calloc (rows, sizeof(gaspi_offset_t));
        for(int i = 0; i < rows; i++)
        {
          /*first row of grid*/
          offset[i] = (start + i) * sizeof(float);
        }
      }
      break;
    case BND_RIGHT:
      if (type == OUTFLOW)
      {
        float *first_ele = grid.getColProxy(cols-2).elemVector();
        gaspi_offset_t start = first_ele - base;
        offset = (gaspi_offset_t *) calloc (rows, sizeof(gaspi_offset_t));
        for(int i = 0; i < rows; i++)
        {
          /*second last row of grid*/
          offset[i] = (start + i) * sizeof(float); 
        }
      }
      else if (type == INFLOW)
      {
        float *first_ele = grid.getColProxy(cols-1).elemVector();
        gaspi_offset_t start = first_ele - base;
        offset = (gaspi_offset_t *) calloc (rows, sizeof(gaspi_offset_t));
        for(int i = 0; i < rows; i++)
        {
          /*Last row of grid*/
          offset[i] = (start + i) * sizeof(float); 
        }
      }
      break;
    case BND_BOTTOM:
      if (type == OUTFLOW)
      {
        float *first_ele = grid.getRowProxy(1).elemVector();
        gaspi_offset_t start = first_ele - base;
        offset = (gaspi_offset_t *) calloc (cols, sizeof(gaspi_offset_t));
        for(int i = 0; i < cols; i++)
        {
          /*second col of grid*/
          offset[i] = (start + (rows)*i) * sizeof(float);
        }
      }
      else if (type == INFLOW)
      {
        float *first_ele = grid.getRowProxy(0).elemVector();
        gaspi_offset_t start = first_ele - base;
        offset = (gaspi_offset_t *) calloc (cols, sizeof(gaspi_offset_t));
        for(int i = 0; i < cols; i++)
        {
          /*first col of grid*/
          offset[i] = (start + (rows)*i) * sizeof(float);
        }
      }
      break;
    case BND_TOP:
      if (type == OUTFLOW)
      {
        float *first_ele = grid.getRowProxy(rows-2).elemVector();
        gaspi_offset_t start = first_ele - base;
        offset = (gaspi_offset_t *) calloc (cols, sizeof(gaspi_offset_t));
        for(int i = 0; i < cols; i++)
        {
          /*second last col of grid*/
          offset[i] = (start + (rows)*i) * sizeof(float); 
        }
      }
      else if (type == INFLOW)
      {
        float *first_ele = grid.getRowProxy(rows-1).elemVector();
        gaspi_offset_t start = first_ele - base;
        offset = (gaspi_offset_t *) calloc (cols, sizeof(gaspi_offset_t));
        for(int i = 0; i < cols; i++)
        {
          /*Last col of grid*/
          offset[i] = (start + (rows)*i) * sizeof(float); 
        }
      }
      break;  
  }
  return offset;
}

void GPI_SWE::exchangeLeftRightGhostLayers ()
{ 
  gaspi_notification_id_t id = l_gpiRank, fid;
  gaspi_notification_t val = 1;
  // send to left, receive from the right:
  if (l_flow.l_leftNeighborRank >= 0)
  {
    for(int j = 0; j < 3; j++)
    {
      for (int i = 0; i < l_flow.l_nleftOutflowOffset; i++)
      {
        l_flow.segment_id_lr[i] = j;
        l_flow.size_lr[i] = sizeof(float);
      }
      val = 1;
      ASSERT(gaspi_write_list_notify(l_flow.l_nleftOutflowOffset,
                                     l_flow.segment_id_lr,
                                     l_flow.l_leftOutflowOffset,
                                     l_flow.l_leftNeighborRank,
                                     l_flow.segment_id_lr,
                                     l_flow.l_rightInflowOffset,
                                     l_flow.size_lr,
                                     j,
                                     id,
                                     val,
                                     0,
                                     GASPI_BLOCK));
      ASSERT(gaspi_wait(0, GASPI_BLOCK));
    }
  }
  
  // send to right, receive from the left:
  if (l_flow.l_rightNeighborRank >= 0)
  {
    for(int j = 0; j < 3; j++)
    {
      for (int i = 0; i < l_flow.l_nrightOutflowOffset; i++)
      {
        l_flow.segment_id_lr[i] = j;
        l_flow.size_lr[i] = sizeof(float);
      }
      val = 1;
      ASSERT(gaspi_write_list_notify(l_flow.l_nrightOutflowOffset,
                                    l_flow.segment_id_lr,
                                    l_flow.l_rightOutflowOffset,
                                    l_flow.l_rightNeighborRank,
                                    l_flow.segment_id_lr,
                                    l_flow.l_leftInflowOffset,
                                    l_flow.size_lr,
                                    j,
                                    id,
                                    val,
                                    0,
                                    GASPI_BLOCK));
      ASSERT(gaspi_wait(0, GASPI_BLOCK));
    }
  }
  if(l_flow.l_rightNeighborRank >= 0)
  {
    /*Wait for a notification from right neighbor for its write on my right inflow*/
    for(int j = 0; j < 3; j++)
    {
      ASSERT(gaspi_notify_waitsome(j, l_flow.l_rightNeighborRank, 1, &fid, GASPI_BLOCK));
      val = 0;
      ASSERT (gaspi_notify_reset(j, fid, &val));
    }
  }
  if(l_flow.l_leftNeighborRank >= 0)
  {
    /*Wait for a notification from right neighbor for its write on my right inflow*/
    for(int j = 0; j < 3; j++)
    {
      ASSERT(gaspi_notify_waitsome(j, l_flow.l_leftNeighborRank, 1, &fid, GASPI_BLOCK));
      val = 0;
      ASSERT (gaspi_notify_reset(j, fid, &val));
    }
  }
}

void GPI_SWE::exchangeBottomTopGhostLayers ()
{
  // send to bottom, receive from the top:
  gaspi_notification_id_t id = l_gpiRank, fid;
  gaspi_notification_t val = 1;
  // send to left, receive from the right:
  if (l_flow.l_bottomNeighborRank >= 0)
  {
    for(int j = 0; j < 3; j++)
    {
      for (int i = 0; i < l_flow.l_nbottomOutflowOffset; i++)
      {
        l_flow.segment_id_bt[i] = j;
        l_flow.size_bt[i] = sizeof(float);
      }
      val = 1;
      ASSERT(gaspi_write_list_notify(l_flow.l_nbottomOutflowOffset,
                                     l_flow.segment_id_bt,
                                     l_flow.l_bottomOutflowOffset,
                                     l_flow.l_bottomNeighborRank,
                                     l_flow.segment_id_bt,
                                     l_flow.l_topInflowOffset,
                                     l_flow.size_bt,
                                     j,
                                     id,
                                     val,
                                     0,
                                     GASPI_BLOCK));
      ASSERT(gaspi_wait(0, GASPI_BLOCK));
    }
  }
  
  // send to top, receive from the bottom:
  if (l_flow.l_topNeighborRank >= 0)
  {
    for(int j = 0; j < 3; j++)
    {
      for (int i = 0; i < l_flow.l_nbottomOutflowOffset; i++)
      {
        l_flow.segment_id_bt[i] = j;
        l_flow.size_bt[i] = sizeof(float);
      }
      val = 1;
      ASSERT(gaspi_write_list_notify(l_flow.l_ntopOutflowOffset,
                                     l_flow.segment_id_bt,
                                     l_flow.l_topOutflowOffset,
                                     l_flow.l_topNeighborRank,
                                     l_flow.segment_id_bt,
                                     l_flow.l_bottomInflowOffset,
                                     l_flow.size_bt,
                                     j,
                                     id,
                                     val,
                                     0,
                                     GASPI_BLOCK));
      ASSERT(gaspi_wait(0, GASPI_BLOCK));
    }
  }
  if(l_flow.l_topNeighborRank >= 0)
  {
    for(int j = 0; j < 3; j++)
    {
      ASSERT(gaspi_notify_waitsome(j, l_flow.l_topNeighborRank, 1, &fid, GASPI_BLOCK));
      val = 0;
      ASSERT (gaspi_notify_reset(j, fid, &val));
    }
  }
  if(l_flow.l_bottomNeighborRank >= 0)
  {
    for(int j = 0; j < 3; j++)
    {
      ASSERT(gaspi_notify_waitsome(j, l_flow.l_bottomNeighborRank, 1, &fid, GASPI_BLOCK));
      val = 0;
      ASSERT (gaspi_notify_reset(j, fid, &val));
    }
  }
}
