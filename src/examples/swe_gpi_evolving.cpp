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

/**
 * Compute the number of block rows from the total number of processes.
 *
 * The number of rows is determined as the square root of the
 * number of processes, if this is a square number;
 * otherwise, we use the largest number that is smaller than the square
 * root and still a divisor of the number of processes.
 *
 * @param numProcs number of process.
 * @return number of block rows
 */
int computeNumberOfBlockRows(int i_numberOfProcesses) {
  int l_numberOfRows = std::sqrt(i_numberOfProcesses);
  while (i_numberOfProcesses % l_numberOfRows != 0) l_numberOfRows--;
  return l_numberOfRows;
}

// Exchanges the left and right ghost layers.
void exchangeLeftRightGhostLayers( flow_info_t &l_flow,
                                   gaspi_rank_t l_gpiRank
                                  );

// Exchanges the bottom and top ghist layers.
void exchangeBottomTopGhostLayers( flow_info_t &l_flow,
                                   gaspi_rank_t l_gpiRank
                                  );

gaspi_return_t recalculateRedistribute(gaspi_rank_t l_gpiRank,
                                       gaspi_rank_t l_numberOfProcesses,
                                       SWE_BathymetryDamBreakScenario &l_scenario,
                                       SWE_Block** l_waveBlock,
                                       float **l_height,
                                       float **l_discharge_hu,
                                       float **l_discharge_hv,
                                       int l_oldRows,
                                       int l_oldCols,
                                       flow_info_t &l_flow,
                                       int l_nX,
                                       int l_nY,
                                       std::shared_ptr<io::Writer>& l_writer,
                                       std::string& l_baseName, int checkpt);

gaspi_offset_t *calculateOffsets(Float2D& grid, BoundaryEdge edge, BoundaryType type);
gaspi_return_t redistributeData(float *l_height, float *l_discharge_hu, float *l_discharge_hv,
                                int l_oldRows, int l_oldCols, int l_gpiRank);

/**
 * Main program for the simulation on a single SWE_WavePropagationBlock or SWE_WaveAccumulationBlock.
 */
int main( int argc, char** argv ) {
  /**
   * Initialization.
   */
  //! MPI Rank of a process.
  gaspi_rank_t l_gpiRank;
  //! number of MPI processes.
  gaspi_rank_t l_numberOfProcesses;
  flow_info_t l_flow;
  float l_t = 0.0;
  unsigned int l_iterations = 0;
  int l_startc = 1;

  // initialize MPI
  gaspi_config_t config;
  ASSERT(gaspi_config_get(&config));
  config.queue_size_max = 1024;
  config.rw_list_elem_max = 1024;
  ASSERT(gaspi_config_set(config));
  ASSERT(gaspi_proc_init(GASPI_BLOCK));

  // determine local MPI rank
  gaspi_proc_rank(&l_gpiRank);
  // determine total number of processes
  gaspi_proc_num(&l_numberOfProcesses);
  std::cout << "started process: " << l_gpiRank;
  // initialize a logger for every MPI process
  tools::Logger::logger.setProcessRank(l_gpiRank);

  // print the welcome message
  tools::Logger::logger.printWelcomeMessage();

  // set current wall clock time within the solver
  tools::Logger::logger.initWallClockTime(time(NULL));
  //print the number of processes
  tools::Logger::logger.printNumberOfProcesses(l_numberOfProcesses);

  // check if the necessary command line input parameters are given
  tools::Args args;
  
  args.addOption("grid-size-x", 'x', "Number of cell in x direction");
  args.addOption("grid-size-y", 'y', "Number of cell in y direction");
  args.addOption("output-basepath", 'o', "Output base file name");
  args.addOption("output-steps-count", 'c', "Number of output time steps");
  
#ifdef ASAGI
  args.addOption("bathymetry-file", 'b', "File containing the bathymetry");
  args.addOption("displacement-file", 'd', "File containing the displacement");
  args.addOption("simul-area-min-x", 0, "Simulation area");
  args.addOption("simul-area-max-x", 0, "Simulation area");
  args.addOption("simul-area-min-y", 0, "Simulation area");
  args.addOption("simul-area-max-y", 0, "Simulation area");
  args.addOption("simul-duration", 0, "Simulation time in seconds");
#endif

  tools::Args::Result ret = args.parse(argc, argv, l_gpiRank == 0);

  switch (ret)
  {
  case tools::Args::Error:
	  exit(EXIT_FAILURE);
  case tools::Args::Help:
	  ASSERT(gaspi_proc_term(GASPI_BLOCK));
	  return 0;
  default:
      break;
  }

  //! total number of grid cell in x- and y-direction.
  int l_nX, l_nY;

  //! l_baseName of the plots.
  std::string l_baseName;

  // read command line parameters
  l_nX = args.getArgument<int>("grid-size-x");
  l_nY = args.getArgument<int>("grid-size-y");
  l_baseName = args.getArgument<std::string>("output-basepath");

  //! number of SWE_Blocks in x- and y-direction.
  int l_blocksX, l_blocksY;

  // determine the layout of MPI-ranks: use l_blocksX*l_blocksY grid blocks
  /*Assume only column splits for simple data redistribution.*/
  l_blocksY = l_numberOfProcesses;
  l_blocksX = l_numberOfProcesses/l_blocksY;

  // print information about the grid
  tools::Logger::logger.printNumberOfCells(l_nX, l_nY);
  tools::Logger::logger.printNumberOfBlocks(l_blocksX, l_blocksY);

  //! local position of each MPI process in x- and y-direction.
  int l_blockPositionX, l_blockPositionY;

  // determine local block coordinates of each SWE_Block
  l_blockPositionX = l_gpiRank / l_blocksY;
  l_blockPositionY = l_gpiRank % l_blocksY;

  #ifdef ASAGI
  /*
   * Pixel node registration used [Cartesian grid]
   * Grid file format: nf = GMT netCDF format (float)  (COARDS-compliant)
   * x_min: -500000 x_max: 6500000 x_inc: 500 name: x nx: 14000
   * y_min: -2500000 y_max: 1500000 y_inc: 500 name: y ny: 8000
   * z_min: -6.48760175705 z_max: 16.1780223846 name: z
   * scale_factor: 1 add_offset: 0
   * mean: 0.00217145586762 stdev: 0.245563641735 rms: 0.245573241263
   */

  //simulation area
  float simulationArea[4];
  simulationArea[0] = args.getArgument<float>("simul-area-min-x");
  simulationArea[1] = args.getArgument<float>("simul-area-max-x");
  simulationArea[2] = args.getArgument<float>("simul-area-min-y");
  simulationArea[3] = args.getArgument<float>("simul-area-max-y");

  float simulationDuration = args.getArgument<float>("simul-duration");

  SWE_AsagiScenario l_scenario(args.getArgument<std::string>("bathymetry-file"), args.getArgument<std::string>("displacement-file"),
                               simulationDuration, simulationArea);
  #else
  // create a simple artificial scenario
  //SWE_RadialDamBreakScenario l_scenario;
  SWE_BathymetryDamBreakScenario l_scenario;
  #endif

  //! number of checkpoints for visualization (at each checkpoint in time, an output file is written).
  int l_numberOfCheckPoints = args.getArgument<int>("output-steps-count");

  //! number of grid cells in x- and y-direction per process.
  int l_nXLocal, l_nYLocal;
  int l_nXNormal, l_nYNormal;

  //! size of a single cell in x- and y-direction
  float l_dX, l_dY;

  // compute local number of cells for each SWE_Block
  l_nXLocal = (l_blockPositionX < l_blocksX-1) ? l_nX/l_blocksX : l_nX - (l_blocksX-1)*(l_nX/l_blocksX);
  l_nYLocal = (l_blockPositionY < l_blocksY-1) ? l_nY/l_blocksY : l_nY - (l_blocksY-1)*(l_nY/l_blocksY);
  l_nXNormal = l_nX/l_blocksX;
  l_nYNormal = l_nY/l_blocksY;

  // compute the size of a single cell
  l_dX = (l_scenario.getBoundaryPos(BND_RIGHT) - l_scenario.getBoundaryPos(BND_LEFT) )/l_nX;
  l_dY = (l_scenario.getBoundaryPos(BND_TOP) - l_scenario.getBoundaryPos(BND_BOTTOM) )/l_nY;

  // print information about the cell size and local number of cells
  tools::Logger::logger.printCellSize(l_dX, l_dY);
  tools::Logger::logger.printNumberOfCellsPerProcess(l_nXLocal, l_nYLocal);

  //! origin of the simulation domain in x- and y-direction
  float l_originX, l_originY;

  // get the origin from the scenario
  l_originX = l_scenario.getBoundaryPos(BND_LEFT) + l_blockPositionX*l_nXNormal*l_dX;;
  l_originY = l_scenario.getBoundaryPos(BND_BOTTOM) + l_blockPositionY*l_nYNormal*l_dY;

  // create a single wave propagation block
  int rows = l_nXLocal+2;
  int cols = l_nYLocal+2;
  float *l_height = new float[rows*cols];
  float *l_discharge_hu = new float[rows*cols];
  float *l_discharge_hv = new float[rows*cols];
  Float2D l_h(rows, cols, l_height);
  Float2D l_hu(rows, cols, l_discharge_hu);
  Float2D l_hv(rows, cols, l_discharge_hv);
  auto l_waveBlock = SWE_Block::getBlockInstance(l_nXLocal,
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
  /*If we are spawned, wait till we receive data from other procs,
  else, initialize the wave propgation block*/
  int spawned = 0;
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
  
  //! time when the simulation ends.
  float l_endSimulation = l_scenario.endSimulation();

  //! checkpoints when output files are written.
  float* l_checkPoints = new float[l_numberOfCheckPoints+1];

  // compute the checkpoints in time
  for(int cp = 0; cp <= l_numberOfCheckPoints; cp++) {
     l_checkPoints[cp] = cp*(l_endSimulation/l_numberOfCheckPoints);
  }

  /*
   * Connect SWE blocks at boundaries
   */
  // left and right boundaries
  if (l_blockPositionX == 0)
    l_waveBlock->setBoundaryType(BND_LEFT, OUTFLOW);
  if (l_blockPositionX == l_blocksX-1)
    l_waveBlock->setBoundaryType(BND_RIGHT, OUTFLOW);

  // bottom and top boundaries
  if (l_blockPositionY == 0)
    l_waveBlock->setBoundaryType(BND_BOTTOM, OUTFLOW);

  if (l_blockPositionY == l_blocksY-1)
    l_waveBlock->setBoundaryType(BND_TOP, OUTFLOW);
  
  l_flow.l_nleftOutflowOffset = l_waveBlock->getWaterHeight().getRows();
  l_flow.l_nleftInflowOffset = l_flow.l_nleftOutflowOffset;
  l_flow.l_nrightOutflowOffset = l_flow.l_nleftOutflowOffset;
  l_flow.l_nrightInflowOffset = l_flow.l_nleftOutflowOffset;
  tools::Logger::logger.cout() << "Number of left right offsets " << l_flow.l_nleftOutflowOffset << std::endl;
  tools::Logger::logger.printString("Connecting SWE blocks at left boundaries.");
  l_flow.l_leftOutflowOffset = calculateOffsets(l_h, BND_LEFT, OUTFLOW);
  l_flow.l_leftInflowOffset = calculateOffsets(l_h, BND_LEFT, INFLOW);
  tools::Logger::logger.printString("Connecting SWE blocks at right boundaries.");
  l_flow.l_rightOutflowOffset = calculateOffsets(l_h, BND_RIGHT, OUTFLOW);
  l_flow.l_rightInflowOffset = calculateOffsets(l_h, BND_RIGHT, INFLOW);

  l_flow.l_nbottomOutflowOffset = l_waveBlock->getWaterHeight().getCols();
  l_flow.l_nbottomInflowOffset = l_flow.l_nbottomOutflowOffset;
  l_flow.l_ntopOutflowOffset = l_flow.l_nbottomOutflowOffset;
  l_flow.l_ntopInflowOffset = l_flow.l_nbottomOutflowOffset;
  tools::Logger::logger.cout() << "Number of top bottom offsets " << l_flow.l_ntopOutflowOffset << std::endl;
  tools::Logger::logger.printString("Connecting SWE blocks at bottom boundaries.");
  l_flow.l_bottomOutflowOffset = calculateOffsets(l_h, BND_BOTTOM, OUTFLOW);
  l_flow.l_bottomInflowOffset = calculateOffsets(l_h, BND_BOTTOM, INFLOW);
  tools::Logger::logger.printString("Connecting SWE blocks at top boundaries.");
  l_flow.l_topOutflowOffset = calculateOffsets(l_h, BND_TOP, OUTFLOW);
  l_flow.l_topInflowOffset = calculateOffsets(l_h, BND_TOP, INFLOW);
  /*
   * The grid is stored column wise in memory:
   *
   *        ************************** . . . **********
   *        *       *  ny+2 *2(ny+2)*         * (ny+1)*
   *        *  ny+1 * +ny+1 * +ny+1 *         * (ny+2)*
   *        *       *       *       *         * +ny+1 *
   *        ************************** . . . **********
   *        *       *       *       *         *       *
   *        .       .       .       .         .       .
   *        .       .       .       .         .       .
   *        .       .       .       .         .       .
   *        *       *       *       *         *       *
   *        ************************** . . . **********
   *        *       *  ny+2 *2(ny+2)*         * (ny+1)*
   *        *   1   *   +1  *   +1  *         * (ny+2)*
   *        *       *       *       *         *   +1  *
   *        ************************** . . . **********
   *        *       *  ny+2 *2(ny+2)*         * (ny+1)*
   *        *   0   *   +0  *   +0  *         * (ny+2)*
   *        *       *       *       *         *   +0  *
   *        ************************** . . . ***********
   *
   *
   *  -> The stride for a row is ny+2, because we have to jump over a whole column
   *     for every row-element. This holds only in the CPU-version, in CUDA a buffer is implemented.
   *     See SWE_BlockCUDA.hh/.cu for details.
   *  -> The stride for a column is 1, because we can access the elements linear in memory.
   */

  //! MPI ranks of the neighbors
  // compute MPI ranks of the neighbour processes
  l_flow.l_leftNeighborRank   = (l_blockPositionX > 0) ? l_gpiRank-l_blocksY : -1;
  l_flow.l_rightNeighborRank  = (l_blockPositionX < l_blocksX-1) ? l_gpiRank+l_blocksY : -1;
  l_flow.l_bottomNeighborRank = (l_blockPositionY > 0) ? l_gpiRank-1 : -1;
  l_flow.l_topNeighborRank    = (l_blockPositionY < l_blocksY-1) ? l_gpiRank+1 : -1;

  // print the MPI grid
  tools::Logger::logger.cout() << "neighbors: "
                     << l_flow.l_leftNeighborRank << " (left), "
                     << l_flow.l_rightNeighborRank << " (right), "
                     << l_flow.l_bottomNeighborRank << " (bottom), "
                     << l_flow.l_topNeighborRank << " (top)" << std::endl;

  // intially exchange ghost and copy layers
  l_flow.segment_id_lr = (gaspi_segment_id_t *) calloc (l_flow.l_nleftOutflowOffset, sizeof(gaspi_segment_id_t));
  l_flow.size_lr = (gaspi_size_t *) calloc (l_flow.l_nleftOutflowOffset, sizeof(gaspi_size_t));
  exchangeLeftRightGhostLayers( l_flow,
                                l_gpiRank
                              );
  
  tools::Logger::logger.cout() << "Exchanged initial left right ghost layers." << std::endl;
  l_flow.segment_id_bt = (gaspi_segment_id_t *) calloc (l_flow.l_nbottomOutflowOffset, sizeof(gaspi_segment_id_t));
  l_flow.size_bt = (gaspi_size_t *) calloc (l_flow.l_nbottomOutflowOffset, sizeof(gaspi_size_t));
  exchangeBottomTopGhostLayers( l_flow,
                                l_gpiRank
                              );
  ASSERT(gaspi_barrier(GASPI_GROUP_ALL, GASPI_BLOCK));
  tools::Logger::logger.cout() << "Exchanged initial top bottom ghost layers." << std::endl;
  // Init fancy progressbar
  tools::ProgressBar progressBar(l_endSimulation, l_gpiRank);

  // write the output at time zero
  tools::Logger::logger.printOutputTime(0);
  progressBar.update(0.);

  std::string l_fileName = generateBaseFileName(l_baseName,l_blockPositionX,l_blockPositionY);
  //boundary size of the ghost layers
  io::BoundarySize l_boundarySize = {{1, 1, 1, 1}};
  //
  // Write zero time step
  /*Synchronize iterations, checkpoints and time steps*/
  std::shared_ptr<io::Writer> l_writer;
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
    l_writer->writeTimeStep( l_waveBlock->getWaterHeight(),
                            l_waveBlock->getDischarge_hu(),
                            l_waveBlock->getDischarge_hv(),
                            (float) l_t);
  }
  /**
   * Simulation.
   */
  // print the start message and reset the wall clock time
  progressBar.clear();
  tools::Logger::logger.printStartMessage();
  tools::Logger::logger.initWallClockTime(time(NULL));

  //! simulation time.
  
  progressBar.update(l_t);
  
  // loop over checkpoints
  for(int c=l_startc; c<=l_numberOfCheckPoints; c++) {

    // do time steps until next checkpoint is reached
    while( l_t < l_checkPoints[c] ) {
      //reset CPU-Communication clock
      tools::Logger::logger.resetClockToCurrentTime("CpuCommunication");

      // exchange ghost and copy layers
      exchangeLeftRightGhostLayers( l_flow,
                                    l_gpiRank
                                  );
      
      exchangeBottomTopGhostLayers( l_flow,
                                    l_gpiRank
                                  );
      ASSERT(gaspi_barrier(GASPI_GROUP_ALL, GASPI_BLOCK));
      // reset the cpu clock
      tools::Logger::logger.resetClockToCurrentTime("Cpu");

      // set values in ghost cells
      l_waveBlock->setGhostLayer();

      // compute numerical flux on each edge
      l_waveBlock->computeNumericalFluxes();

      //! maximum allowed time step width within a block.
      float l_maxTimeStepWidth = l_waveBlock->getMaxTimestep();

      // update the cpu time in the logger
      tools::Logger::logger.updateTime("Cpu");

      //! maximum allowed time steps of all blocks
      float l_maxTimeStepWidthGlobal;

      // determine smallest time step of all blocks
      gaspi_allreduce(&l_maxTimeStepWidth, &l_maxTimeStepWidthGlobal, 1,
                      GASPI_OP_MIN, GASPI_TYPE_FLOAT, GASPI_GROUP_ALL, GASPI_BLOCK);

      // reset the cpu time
      tools::Logger::logger.resetClockToCurrentTime("Cpu");

      // update the cell values
      l_waveBlock->updateUnknowns(l_maxTimeStepWidthGlobal);

      // update the cpu and CPU-communication time in the logger
      tools::Logger::logger.updateTime("Cpu");
      tools::Logger::logger.updateTime("CpuCommunication");

      // update simulation time with time step width.
      l_t += l_maxTimeStepWidthGlobal;
      l_iterations++;

      // print the current simulation time
      progressBar.clear();
      tools::Logger::logger.printSimulationTime(l_t);
      progressBar.update(l_t);
    }

    // print current simulation time
    progressBar.clear();
    tools::Logger::logger.printOutputTime(l_t);
    progressBar.update(l_t);

    // write output
    l_writer->writeTimeStep( l_waveBlock->getWaterHeight(),
                            l_waveBlock->getDischarge_hu(),
                            l_waveBlock->getDischarge_hv(),
                            l_t);
    /*We want to now add more resources to the computation.
    We request l_numberOfProcesses of additional processes.*/
    ASSERT(gaspi_segment_delete(0));
    ASSERT(gaspi_segment_delete(1));
    ASSERT(gaspi_segment_delete(2));
    if(c+1 <= l_numberOfCheckPoints)
    {
      tools::Logger::logger.cout() << "Allocating " << l_numberOfProcesses 
                    << " procs with group name " << argv[0] << std::endl;
      ASSERT(gaspi_proc_alloc_request(l_numberOfProcesses,
                                      &l_gpiRank, 
                                      &l_numberOfProcesses,
                                      argc, argv));
      tools::Logger::logger.cout() << "We now have " << l_numberOfProcesses << " processes."
                                    << std::endl;;
      /*Now we recalculate all metadata and redistribute data.*/
      ASSERT(recalculateRedistribute(l_gpiRank,
                                    l_numberOfProcesses,
                                    l_scenario,
                                    &l_waveBlock,
                                    &l_height,
                                    &l_discharge_hu,
                                    &l_discharge_hv,
                                    l_nXLocal,
                                    l_nYLocal,
                                    l_flow,
                                    l_nX,
                                    l_nY,
                                    l_writer,
                                    l_baseName, c));
      l_nXLocal = l_waveBlock->getNx();
      l_nYLocal = l_waveBlock->getNy();
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
    }
  }

  /**
   * Finalize.
   */
#ifdef ASAGI
  // Free ASAGI resources
  l_scenario.deleteGrids();
#endif

  progressBar.clear();

  // write the statistics message
  tools::Logger::logger.printStatisticsMessage();

  // print the cpu time
  tools::Logger::logger.printTime("Cpu", "CPU time");

  // print CPU + Communication time
  tools::Logger::logger.printTime("CpuCommunication", "CPU + Communication time");

  // print the wall clock time (includes plotting)
  tools::Logger::logger.printWallClockTime(time(NULL));

  // printer iteration counter
  tools::Logger::logger.printIterationsDone(l_iterations);

  // print the finish message
  tools::Logger::logger.printFinishMessage();

  // Dispose of the SWE block!
  delete l_waveBlock;

  // finalize MPI execution
  ASSERT(gaspi_proc_term(GASPI_BLOCK));

  return 0;
}

gaspi_return_t redistributeData(float *l_height, float *l_discharge_hu, float *l_discharge_hv,
                                int l_oldRows, int l_oldCols, gaspi_rank_t l_gpiRank)
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

gaspi_return_t recalculateRedistribute(gaspi_rank_t l_gpiRank,
                                       gaspi_rank_t l_numberOfProcesses,
                                       SWE_BathymetryDamBreakScenario &l_scenario,
                                       SWE_Block** l_waveBlock,
                                       float **l_height,
                                       float **l_discharge_hu,
                                       float **l_discharge_hv,
                                       int l_oldRows,
                                       int l_oldCols,
                                       flow_info_t &l_flow,
                                       int l_nX,
                                       int l_nY,
                                       std::shared_ptr<io::Writer>& l_writer,
                                       std::string& l_baseName, int checkpoint)
{
  tools::Logger::logger.setProcessRank(l_gpiRank);
  int l_blocksX, l_blocksY;

  /*We assume splitting the x-axis according to the available
  processes for ease of data redistribution.*/
  l_blocksY = l_numberOfProcesses;
  l_blocksX = l_numberOfProcesses/l_blocksY;

  int l_blockPositionX, l_blockPositionY;

  // determine local block coordinates of each SWE_Block
  l_blockPositionX = l_gpiRank / l_blocksY;
  l_blockPositionY = l_gpiRank % l_blocksY;

  //! number of grid cells in x- and y-direction per process.
  int l_nXLocal, l_nYLocal;
  int l_nXNormal, l_nYNormal;

  //! size of a single cell in x- and y-direction
  float l_dX, l_dY;

  // compute local number of cells for each SWE_Block
  l_nXLocal = (l_blockPositionX < l_blocksX-1) ? l_nX/l_blocksX : l_nX - (l_blocksX-1)*(l_nX/l_blocksX);
  l_nYLocal = (l_blockPositionY < l_blocksY-1) ? l_nY/l_blocksY : l_nY - (l_blocksY-1)*(l_nY/l_blocksY);
  l_nXNormal = l_nX/l_blocksX;
  l_nYNormal = l_nY/l_blocksY;

  // compute the size of a single cell
  l_dX = (l_scenario.getBoundaryPos(BND_RIGHT) - l_scenario.getBoundaryPos(BND_LEFT) )/l_nX;
  l_dY = (l_scenario.getBoundaryPos(BND_TOP) - l_scenario.getBoundaryPos(BND_BOTTOM) )/l_nY;
  float l_originX, l_originY;

  // get the origin from the scenario
  l_originX = l_scenario.getBoundaryPos(BND_LEFT) + l_blockPositionX*l_nXNormal*l_dX;;
  l_originY = l_scenario.getBoundaryPos(BND_BOTTOM) + l_blockPositionY*l_nYNormal*l_dY;

  // create a single wave propagation block
  /*We create new blocks, redistribute data, delete old blocks and
  assign new block pointers to passed pointers.
  */
  int rows = l_nXLocal+2;
  int cols = l_nYLocal+2;
  
  float *new_height = new float[rows*cols];
  float *new_discharge_hu = new float[rows*cols];
  float *new_discharge_hv = new float[rows*cols];

  Float2D l_h(rows, cols, new_height);
  Float2D l_hu(rows, cols, new_discharge_hu);
  Float2D l_hv(rows, cols, new_discharge_hv);
  delete(*l_waveBlock);
  *l_waveBlock = SWE_Block::getBlockInstance(l_nXLocal,
                                             l_nYLocal, 
                                             l_dX, 
                                             l_dY,
                                             l_h,
                                             l_hu,
                                             l_hv);
  tools::Logger::logger.cout() << "Waiting for segments to get created " << std::endl;
  std::cout.flush();
  ASSERT(gaspi_segment_use(0, (gaspi_pointer_t) new_height,
                           (rows) * (cols) * sizeof(float),
                           GASPI_GROUP_ALL, GASPI_BLOCK, 0));
  ASSERT(gaspi_segment_use(1, (gaspi_pointer_t) new_discharge_hu,
                           (rows) * (cols) * sizeof(float),
                           GASPI_GROUP_ALL, GASPI_BLOCK, 0));
  ASSERT(gaspi_segment_use(2, (gaspi_pointer_t) new_discharge_hv,
                           (rows) * (cols) * sizeof(float),
                           GASPI_GROUP_ALL, GASPI_BLOCK, 0));
  
  SWE_Block* l_newWaveBlock = *l_waveBlock;
  l_newWaveBlock->initScenario(l_originX, l_originY, l_scenario, true);
  tools::Logger::logger.cout() << "Waiting for segments to get created " << std::endl;
  std::cout.flush();
  ASSERT(gaspi_barrier(GASPI_GROUP_ALL, GASPI_BLOCK));
  tools::Logger::logger.cout() << "Segments have been created with rows " 
                               << rows << " and cols " << cols << std::endl;
  /*Now all procs have created new memory segments.
  We want to begin redistributing data*/
  ASSERT(redistributeData(*l_height, *l_discharge_hu, *l_discharge_hv,
                          l_oldRows, l_oldCols, l_gpiRank));
  tools::Logger::logger.cout() << "Data has been redistributed " << std::endl;
  /*After redistributing data, we can delete old pointers.*/
  free(*l_height);
  free(*l_discharge_hu);
  free(*l_discharge_hv);
  *l_height = new_height;
  *l_discharge_hu = new_discharge_hu;
  *l_discharge_hv = new_discharge_hv;
  
  if (l_blockPositionX == 0)
    l_newWaveBlock->setBoundaryType(BND_LEFT, OUTFLOW);
  if (l_blockPositionX == l_blocksX-1)
    l_newWaveBlock->setBoundaryType(BND_RIGHT, OUTFLOW);

  // bottom and top boundaries
  if (l_blockPositionY == 0)
    l_newWaveBlock->setBoundaryType(BND_BOTTOM, OUTFLOW);

  if (l_blockPositionY == l_blocksY-1)
    l_newWaveBlock->setBoundaryType(BND_TOP, OUTFLOW);
  
  l_flow.l_nleftOutflowOffset = l_newWaveBlock->getWaterHeight().getRows();
  l_flow.l_nleftInflowOffset = l_flow.l_nleftOutflowOffset;
  l_flow.l_nrightOutflowOffset = l_flow.l_nleftOutflowOffset;
  l_flow.l_nrightInflowOffset = l_flow.l_nleftOutflowOffset;
  tools::Logger::logger.cout() << "Number of left right offsets " << l_flow.l_nleftOutflowOffset << std::endl;
  tools::Logger::logger.printString("Connecting SWE blocks at left boundaries.");
  free(l_flow.l_leftOutflowOffset);
  free(l_flow.l_leftInflowOffset);
  l_flow.l_leftOutflowOffset = calculateOffsets(l_h, BND_LEFT, OUTFLOW);
  l_flow.l_leftInflowOffset = calculateOffsets(l_h, BND_LEFT, INFLOW);
  tools::Logger::logger.printString("Connecting SWE blocks at right boundaries.");
  free(l_flow.l_rightOutflowOffset);
  free(l_flow.l_rightInflowOffset);
  l_flow.l_rightOutflowOffset = calculateOffsets(l_h, BND_RIGHT, OUTFLOW);
  l_flow.l_rightInflowOffset = calculateOffsets(l_h, BND_RIGHT, INFLOW);

  l_flow.l_nbottomOutflowOffset = l_newWaveBlock->getWaterHeight().getCols();
  l_flow.l_nbottomInflowOffset = l_flow.l_nbottomOutflowOffset;
  l_flow.l_ntopOutflowOffset = l_flow.l_nbottomOutflowOffset;
  l_flow.l_ntopInflowOffset = l_flow.l_nbottomOutflowOffset;
  tools::Logger::logger.cout() << "Number of top bottom offsets " << l_flow.l_ntopOutflowOffset << std::endl;
  tools::Logger::logger.printString("Connecting SWE blocks at bottom boundaries.");
  free(l_flow.l_bottomOutflowOffset);
  free(l_flow.l_bottomInflowOffset);
  l_flow.l_bottomOutflowOffset = calculateOffsets(l_h, BND_BOTTOM, OUTFLOW);
  l_flow.l_bottomInflowOffset = calculateOffsets(l_h, BND_BOTTOM, INFLOW);
  tools::Logger::logger.printString("Connecting SWE blocks at top boundaries.");
  free(l_flow.l_topOutflowOffset);
  free(l_flow.l_topInflowOffset);
  l_flow.l_topOutflowOffset = calculateOffsets(l_h, BND_TOP, OUTFLOW);
  l_flow.l_topInflowOffset = calculateOffsets(l_h, BND_TOP, INFLOW);

  // compute MPI ranks of the neighbour processes
  l_flow.l_leftNeighborRank   = (l_blockPositionX > 0) ? l_gpiRank-l_blocksY : -1;
  l_flow.l_rightNeighborRank  = (l_blockPositionX < l_blocksX-1) ? l_gpiRank+l_blocksY : -1;
  l_flow.l_bottomNeighborRank = (l_blockPositionY > 0) ? l_gpiRank-1 : -1;
  l_flow.l_topNeighborRank    = (l_blockPositionY < l_blocksY-1) ? l_gpiRank+1 : -1;

  // print the MPI grid
  tools::Logger::logger.cout() << "neighbors: "
                     << l_flow.l_leftNeighborRank << " (left), "
                     << l_flow.l_rightNeighborRank << " (right), "
                     << l_flow.l_bottomNeighborRank << " (bottom), "
                     << l_flow.l_topNeighborRank << " (top)" << std::endl;

  // intially exchange ghost and copy layers
  free(l_flow.segment_id_lr);
  free(l_flow.size_lr);
  l_flow.segment_id_lr = (gaspi_segment_id_t *) calloc (l_flow.l_nleftOutflowOffset, sizeof(gaspi_segment_id_t));
  l_flow.size_lr = (gaspi_size_t *) calloc (l_flow.l_nleftOutflowOffset, sizeof(gaspi_size_t));
  exchangeLeftRightGhostLayers( l_flow,
                                l_gpiRank
                              );
  
  tools::Logger::logger.cout() << "Exchanged initial left right ghost layers." << std::endl;
  free(l_flow.segment_id_bt);
  free(l_flow.size_bt);
  l_flow.segment_id_bt = (gaspi_segment_id_t *) calloc (l_flow.l_nbottomOutflowOffset, sizeof(gaspi_segment_id_t));
  l_flow.size_bt = (gaspi_size_t *) calloc (l_flow.l_nbottomOutflowOffset, sizeof(gaspi_size_t));
  exchangeBottomTopGhostLayers( l_flow,
                                l_gpiRank
                              );
  ASSERT(gaspi_barrier(GASPI_GROUP_ALL, GASPI_BLOCK));
  tools::Logger::logger.cout() << "Exchanged initial top bottom ghost layers." << std::endl;
  io::BoundarySize l_boundarySize = {{1, 1, 1, 1}};
  std::string l_fileName = generateBaseFileName(l_baseName,l_blockPositionX,l_blockPositionY);
  l_writer = io::Writer::createWriterInstance(
          l_fileName,
          l_newWaveBlock->getBathymetry(),
          l_boundarySize,
          l_nXLocal, l_nYLocal,
          l_dX, l_dY,
          l_blockPositionX*l_nXLocal, l_blockPositionY*l_nYLocal,
          l_originX, l_originY, 0,
          checkpoint);
  return GASPI_SUCCESS;
}

gaspi_offset_t *calculateOffsets(Float2D& grid, BoundaryEdge edge, BoundaryType type)
{
  int rows = grid.getRows(); /*nY*/
  int cols = grid.getCols(); /*nX*/
  float *base = grid.elemVector();
  gaspi_offset_t *offset = NULL;
  //tools::Logger::logger.cout() << "rows = " << rows << " cols = " << cols << std::endl;
  switch(edge)
  {
    case BND_LEFT:
      if (type == OUTFLOW)
      {
        float *first_ele = grid.getColProxy(1).elemVector();
        gaspi_offset_t start = first_ele - base;
        offset = (gaspi_offset_t *) calloc (rows, sizeof(gaspi_offset_t));
        //tools::Logger::logger.cout() << "left outflow offsets = " << std::endl;
        for(int i = 0; i < rows; i++)
        {
          /*second row of grid*/
          offset[i] = (start + i) * sizeof(float);
          //std::cout << offset[i] << "\t"; 
        }
      }
      else if (type == INFLOW)
      {
        float *first_ele = grid.getColProxy(0).elemVector();
        gaspi_offset_t start = first_ele - base;
        offset = (gaspi_offset_t *) calloc (rows, sizeof(gaspi_offset_t));
        //tools::Logger::logger.cout() << "left inflow offsets = " << std::endl;
        for(int i = 0; i < rows; i++)
        {
          /*first row of grid*/
          offset[i] = (start + i) * sizeof(float);
          //std::cout << offset[i] << "\t"; 
        }
      }
      break;
    case BND_RIGHT:
      if (type == OUTFLOW)
      {
        float *first_ele = grid.getColProxy(cols-2).elemVector();
        gaspi_offset_t start = first_ele - base;
        offset = (gaspi_offset_t *) calloc (rows, sizeof(gaspi_offset_t));
        //tools::Logger::logger.cout() << "right outflow offsets = " << std::endl;
        for(int i = 0; i < rows; i++)
        {
          /*second last row of grid*/
          offset[i] = (start + i) * sizeof(float); 
          //std::cout << offset[i] << "\t";
        }
      }
      else if (type == INFLOW)
      {
        float *first_ele = grid.getColProxy(cols-1).elemVector();
        gaspi_offset_t start = first_ele - base;
        offset = (gaspi_offset_t *) calloc (rows, sizeof(gaspi_offset_t));
        //tools::Logger::logger.cout() << "right inflow offsets = " << std::endl;
        for(int i = 0; i < rows; i++)
        {
          /*Last row of grid*/
          offset[i] = (start + i) * sizeof(float); 
          //std::cout << offset[i] << "\t";
        }
      }
      break;
    case BND_BOTTOM:
      if (type == OUTFLOW)
      {
        float *first_ele = grid.getRowProxy(1).elemVector();
        gaspi_offset_t start = first_ele - base;
        offset = (gaspi_offset_t *) calloc (cols, sizeof(gaspi_offset_t));
        //tools::Logger::logger.cout() << "bottom outflow offsets = " << std::endl;
        for(int i = 0; i < cols; i++)
        {
          /*second col of grid*/
          offset[i] = (start + (rows)*i) * sizeof(float);
          //std::cout << offset[i] << "\t"; 
        }
        //tools::Logger::logger.cout() << std::endl;
      }
      else if (type == INFLOW)
      {
        float *first_ele = grid.getRowProxy(0).elemVector();
        gaspi_offset_t start = first_ele - base;
        offset = (gaspi_offset_t *) calloc (cols, sizeof(gaspi_offset_t));
        //tools::Logger::logger.cout() << "bottom inflow offsets = " << std::endl;
        for(int i = 0; i < cols; i++)
        {
          /*first col of grid*/
          offset[i] = (start + (rows)*i) * sizeof(float);
          //std::cout << offset[i] << "\t"; 
        }
        //tools::Logger::logger.cout() << std::endl;
      }
      break;
    case BND_TOP:
      if (type == OUTFLOW)
      {
        float *first_ele = grid.getRowProxy(rows-2).elemVector();
        gaspi_offset_t start = first_ele - base;
        offset = (gaspi_offset_t *) calloc (cols, sizeof(gaspi_offset_t));
        //tools::Logger::logger.cout() << "top outflow offsets = " << std::endl;
        for(int i = 0; i < cols; i++)
        {
          /*second last col of grid*/
          offset[i] = (start + (rows)*i) * sizeof(float); 
          //std::cout << offset[i] << "\t";
        }
        //tools::Logger::logger.cout() << std::endl;
      }
      else if (type == INFLOW)
      {
        float *first_ele = grid.getRowProxy(rows-1).elemVector();
        gaspi_offset_t start = first_ele - base;
        offset = (gaspi_offset_t *) calloc (cols, sizeof(gaspi_offset_t));
        //tools::Logger::logger.cout() << "top inflow offsets = " << std::endl;
        for(int i = 0; i < cols; i++)
        {
          /*Last col of grid*/
          offset[i] = (start + (rows)*i) * sizeof(float); 
          //std::cout << offset[i] << "\t";
        }
        //tools::Logger::logger.cout() << std::endl;
      }
      break;  
  }
  return offset;
}

void exchangeLeftRightGhostLayers( flow_info_t &l_flow,
                                   gaspi_rank_t l_gpiRank
                                  )
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
      //tools::Logger::logger.cout() << id << ": sent segment " << j << std::endl;
    }
    //tools::Logger::logger.cout() << "sent data to my left neighbor" << std::endl;
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
      //tools::Logger::logger.cout() << id << ": sent segment " << j << std::endl;
    }
    //tools::Logger::logger.cout() << "sent data to my right neighbor" << std::endl;
  }
  if(l_flow.l_rightNeighborRank >= 0)
  {
    /*Wait for a notification from right neighbor for its write on my right inflow*/
    for(int j = 0; j < 3; j++)
    {
      ASSERT(gaspi_notify_waitsome(j, l_flow.l_rightNeighborRank, 1, &fid, GASPI_BLOCK));
      val = 0;
      ASSERT (gaspi_notify_reset(j, fid, &val));
      //tools::Logger::logger.cout() << id << ": received segment " << j << std::endl;
    }
    //tools::Logger::logger.cout() << "received data from my right neighbor" << std::endl;
  }
  if(l_flow.l_leftNeighborRank >= 0)
  {
    /*Wait for a notification from right neighbor for its write on my right inflow*/
    for(int j = 0; j < 3; j++)
    {
      ASSERT(gaspi_notify_waitsome(j, l_flow.l_leftNeighborRank, 1, &fid, GASPI_BLOCK));
      val = 0;
      ASSERT (gaspi_notify_reset(j, fid, &val));
      //tools::Logger::logger.cout() << id << ": received segment " << j << std::endl;
    }
    //tools::Logger::logger.cout() << "received data from my left neighbor" << std::endl;
  }
}

void exchangeBottomTopGhostLayers( flow_info_t &l_flow,
                                   gaspi_rank_t l_gpiRank
                                  )
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
    //tools::Logger::logger.cout() << "sent data to my bottom neighbor" << std::endl;
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
    //tools::Logger::logger.cout() << "sent data to my top neighbor" << std::endl;
  }
  if(l_flow.l_topNeighborRank >= 0)
  {
    for(int j = 0; j < 3; j++)
    {
      ASSERT(gaspi_notify_waitsome(j, l_flow.l_topNeighborRank, 1, &fid, GASPI_BLOCK));
      val = 0;
      ASSERT (gaspi_notify_reset(j, fid, &val));
    }
    //tools::Logger::logger.cout() << "received data from my top neighbor" << std::endl;
  }
  if(l_flow.l_bottomNeighborRank >= 0)
  {
    for(int j = 0; j < 3; j++)
    {
      ASSERT(gaspi_notify_waitsome(j, l_flow.l_bottomNeighborRank, 1, &fid, GASPI_BLOCK));
      val = 0;
      ASSERT (gaspi_notify_reset(j, fid, &val));
    }
    //tools::Logger::logger.cout() << "received data from my bottom neighbor" << std::endl;
  }
}
