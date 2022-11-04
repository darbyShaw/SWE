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
void exchangeLeftRightGhostLayers( const int l_leftNeighborRank,
                                   const int l_nleftInflowOffset, 
                                   gaspi_offset_t *l_leftInflowOffset,
                                   const int l_nleftOutflowOffset,
                                   gaspi_offset_t *l_leftOutflowOffset,
                                   const int l_rightNeighborRank,
                                   const int l_nrightInflowOffset,
                                   gaspi_offset_t *l_rightInflowOffset,
                                   int l_nrightOutflowOffset,
                                   gaspi_offset_t *l_rightOutflowOffset,
                                   gaspi_segment_id_t *segment_id,
                                   gaspi_size_t *size,
                                   gaspi_rank_t l_gpiRank
                                  );

// Exchanges the bottom and top ghist layers.
void exchangeBottomTopGhostLayers( const int l_bottomNeighborRank,
                                   const int l_nbottomInflowOffset, 
                                   gaspi_offset_t *l_bottomInflowOffset,
                                   const int l_nbottomOutflowOffset,
                                   gaspi_offset_t *l_bottomOutflowOffset,
                                   const int l_topNeighborRank,
                                   const int l_ntopInflowOffset,
                                   gaspi_offset_t *l_topInflowOffset,
                                   int l_ntopOutflowOffset,
                                   gaspi_offset_t *l_topOutflowOffset,
                                   gaspi_segment_id_t *segment_id,
                                   gaspi_size_t *size,
                                   gaspi_rank_t l_gpiRank
                                  );

gaspi_offset_t *calculateOffsets(Float2D& grid, BoundaryEdge edge, BoundaryType type);

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
  l_blocksY = computeNumberOfBlockRows(l_numberOfProcesses);
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
  Float2D l_h(l_nXLocal+2, l_nYLocal+2);
  Float2D l_hu(l_nXLocal+2, l_nYLocal+2);
  Float2D l_hv(l_nXLocal+2, l_nYLocal+2);
  auto l_waveBlock = SWE_Block::getBlockInstance(l_nXLocal,
                                                 l_nYLocal, 
                                                 l_dX, 
                                                 l_dY,
                                                 l_h,
                                                 l_hu,
                                                 l_hv);
  
  //Bind the block to a gaspi segment
  float *height = l_h.elemVector();
  float *discharge_hu = l_hu.elemVector();
  float *discharge_hv = l_hv.elemVector();

  ASSERT(gaspi_segment_use(0, (gaspi_pointer_t) height,
                           l_nXLocal * l_nYLocal * sizeof(float),
                           GASPI_GROUP_ALL, GASPI_BLOCK, 0));
  ASSERT(gaspi_segment_use(1, (gaspi_pointer_t) discharge_hu,
                           l_nXLocal * l_nYLocal * sizeof(float),
                           GASPI_GROUP_ALL, GASPI_BLOCK, 0));
  ASSERT(gaspi_segment_use(2, (gaspi_pointer_t) discharge_hv,
                           l_nXLocal * l_nYLocal * sizeof(float),
                           GASPI_GROUP_ALL, GASPI_BLOCK, 0));

  // initialize the wave propgation block
  l_waveBlock->initScenario(l_originX, l_originY, l_scenario, true);

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
  
  int l_nleftOutflowOffset = l_waveBlock->getWaterHeight().getRows();
  int l_nleftInflowOffset = l_nleftOutflowOffset;
  int l_nrightOutflowOffset = l_nleftOutflowOffset;
  int l_nrightInflowOffset = l_nleftOutflowOffset;
  tools::Logger::logger.cout() << "Number of left right offsets " << l_nleftOutflowOffset << std::endl;
  tools::Logger::logger.printString("Connecting SWE blocks at left boundaries.");
  gaspi_offset_t *l_leftOutflowOffset = calculateOffsets(l_h, BND_LEFT, OUTFLOW);
  gaspi_offset_t *l_leftInflowOffset = calculateOffsets(l_h, BND_LEFT, INFLOW);
  tools::Logger::logger.printString("Connecting SWE blocks at right boundaries.");
  gaspi_offset_t *l_rightOutflowOffset = calculateOffsets(l_h, BND_RIGHT, OUTFLOW);
  gaspi_offset_t *l_rightInflowOffset = calculateOffsets(l_h, BND_RIGHT, INFLOW);

  int l_nbottomOutflowOffset = l_waveBlock->getWaterHeight().getCols();
  int l_nbottomInflowOffset = l_nbottomOutflowOffset;
  int l_ntopOutflowOffset = l_nbottomOutflowOffset;
  int l_ntopInflowOffset = l_nbottomOutflowOffset;
  tools::Logger::logger.cout() << "Number of top bottom offsets " << l_ntopOutflowOffset << std::endl;
  tools::Logger::logger.printString("Connecting SWE blocks at bottom boundaries.");
  gaspi_offset_t *l_bottomOutflowOffset = calculateOffsets(l_h, BND_BOTTOM, OUTFLOW);
  gaspi_offset_t *l_bottomInflowOffset = calculateOffsets(l_h, BND_BOTTOM, INFLOW);
  tools::Logger::logger.printString("Connecting SWE blocks at top boundaries.");
  gaspi_offset_t *l_topOutflowOffset = calculateOffsets(l_h, BND_TOP, OUTFLOW);
  gaspi_offset_t *l_topInflowOffset = calculateOffsets(l_h, BND_TOP, INFLOW);
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
  int l_leftNeighborRank, l_rightNeighborRank, l_bottomNeighborRank, l_topNeighborRank;

  // compute MPI ranks of the neighbour processes
  l_leftNeighborRank   = (l_blockPositionX > 0) ? l_gpiRank-l_blocksY : -1;
  l_rightNeighborRank  = (l_blockPositionX < l_blocksX-1) ? l_gpiRank+l_blocksY : -1;
  l_bottomNeighborRank = (l_blockPositionY > 0) ? l_gpiRank-1 : -1;
  l_topNeighborRank    = (l_blockPositionY < l_blocksY-1) ? l_gpiRank+1 : -1;

  // print the MPI grid
  tools::Logger::logger.cout() << "neighbors: "
                     << l_leftNeighborRank << " (left), "
                     << l_rightNeighborRank << " (right), "
                     << l_bottomNeighborRank << " (bottom), "
                     << l_topNeighborRank << " (top)" << std::endl;

  // intially exchange ghost and copy layers
  gaspi_segment_id_t *segment_id_lr = NULL;
  gaspi_size_t *size_lr = NULL;
  segment_id_lr = (gaspi_segment_id_t *) calloc (l_nleftOutflowOffset, sizeof(gaspi_segment_id_t));
  size_lr = (gaspi_size_t *) calloc (l_nleftOutflowOffset, sizeof(gaspi_size_t));
  exchangeLeftRightGhostLayers( l_leftNeighborRank,
                                l_nleftInflowOffset, 
                                l_leftInflowOffset,
                                l_nleftOutflowOffset,
                                l_leftOutflowOffset,
                                l_rightNeighborRank,
                                l_nrightInflowOffset,
                                l_rightInflowOffset,
                                l_nrightOutflowOffset,
                                l_rightOutflowOffset,
                                segment_id_lr,
                                size_lr,
                                l_gpiRank
                              );
  ASSERT(gaspi_barrier(GASPI_GROUP_ALL, GASPI_BLOCK));
  tools::Logger::logger.cout() << "Exchanged initial left right ghost layers." << std::endl;
  gaspi_segment_id_t *segment_id_bt = NULL;
  gaspi_size_t *size_bt = NULL;
  segment_id_bt = (gaspi_segment_id_t *) calloc (l_nbottomOutflowOffset, sizeof(gaspi_segment_id_t));
  size_bt = (gaspi_size_t *) calloc (l_nbottomOutflowOffset, sizeof(gaspi_size_t));
  exchangeBottomTopGhostLayers( l_bottomNeighborRank, 
                                l_nbottomInflowOffset,
                                l_bottomInflowOffset,
                                l_nbottomOutflowOffset,
                                l_bottomOutflowOffset,
                                l_topNeighborRank,
                                l_ntopInflowOffset,
                                l_topInflowOffset,
                                l_ntopOutflowOffset,
                                l_topOutflowOffset,
                                segment_id_bt,
                                size_bt,
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
  /*
#ifdef WRITENETCDF
  //construct a NetCdfWriter
  io::NetCdfWriter l_writer( l_fileName,
		  l_waveBlock->getBathymetry(),
		  l_boundarySize,
		  l_nXLocal, l_nYLocal,
		  l_dX, l_dY,
          l_originX, l_originY );
#else
  // Construct a VtkWriter
  io::VtkWriter l_writer( l_fileName,
		  l_waveBlock->getBathymetry(),
		  l_boundarySize,
		  l_nXLocal, l_nYLocal,
		  l_dX, l_dY,
		  l_blockPositionX*l_nXLocal, l_blockPositionY*l_nYLocal );
#endif
  */
  auto l_writer = io::Writer::createWriterInstance(
          l_fileName,
          l_waveBlock->getBathymetry(),
          l_boundarySize,
          l_nXLocal, l_nYLocal,
          l_dX, l_dY,
          l_blockPositionX*l_nXLocal, l_blockPositionY*l_nYLocal,
          l_originX, l_originY,
          0);
  //
  // Write zero time step
  l_writer->writeTimeStep( l_waveBlock->getWaterHeight(),
                          l_waveBlock->getDischarge_hu(),
                          l_waveBlock->getDischarge_hv(),
                          (float) 0.);
  /**
   * Simulation.
   */
  // print the start message and reset the wall clock time
  progressBar.clear();
  tools::Logger::logger.printStartMessage();
  tools::Logger::logger.initWallClockTime(time(NULL));

  //! simulation time.
  float l_t = 0.0;
  progressBar.update(l_t);

  unsigned int l_iterations = 0;

  // loop over checkpoints
  for(int c=1; c<=l_numberOfCheckPoints; c++) {

    // do time steps until next checkpoint is reached
    while( l_t < l_checkPoints[c] ) {
      //reset CPU-Communication clock
      tools::Logger::logger.resetClockToCurrentTime("CpuCommunication");

      // exchange ghost and copy layers
      exchangeLeftRightGhostLayers( l_leftNeighborRank,
                                    l_nleftInflowOffset, 
                                    l_leftInflowOffset,
                                    l_nleftOutflowOffset,
                                    l_leftOutflowOffset,
                                    l_rightNeighborRank,
                                    l_nrightInflowOffset,
                                    l_rightInflowOffset,
                                    l_nrightOutflowOffset,
                                    l_rightOutflowOffset,
                                    segment_id_lr,
                                    size_lr,
                                    l_gpiRank
                                  );
      ASSERT(gaspi_barrier(GASPI_GROUP_ALL, GASPI_BLOCK));
      exchangeBottomTopGhostLayers( l_bottomNeighborRank, 
                                    l_nbottomInflowOffset,
                                    l_bottomInflowOffset,
                                    l_nbottomOutflowOffset,
                                    l_bottomOutflowOffset,
                                    l_topNeighborRank,
                                    l_ntopInflowOffset,
                                    l_topInflowOffset,
                                    l_ntopOutflowOffset,
                                    l_topOutflowOffset,
                                    segment_id_bt,
                                    size_bt,
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
          offset[i] = start + i;
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
          offset[i] = start + i;
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
          offset[i] = start + i; 
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
          offset[i] = start + i; 
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
          offset[i] = start + (rows)*i;
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
          offset[i] = start + (rows)*i;
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
          /*second last row of grid*/
          offset[i] = start + (rows)*i; 
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
          /*Last row of grid*/
          offset[i] = start + (rows)*i; 
          std::cout << offset[i] << "\t";
        }
        //tools::Logger::logger.cout() << std::endl;
      }
      break;  
  }
  return offset;
}

void exchangeLeftRightGhostLayers( const int l_leftNeighborRank,
                                   const int l_nleftInflowOffset, 
                                   gaspi_offset_t *l_leftInflowOffset,
                                   const int l_nleftOutflowOffset,
                                   gaspi_offset_t *l_leftOutflowOffset,
                                   const int l_rightNeighborRank,
                                   const int l_nrightInflowOffset,
                                   gaspi_offset_t *l_rightInflowOffset,
                                   int l_nrightOutflowOffset,
                                   gaspi_offset_t *l_rightOutflowOffset,
                                   gaspi_segment_id_t *segment_id,
                                   gaspi_size_t *size,
                                   gaspi_rank_t l_gpiRank
                                  )
{ 
  gaspi_notification_id_t id = l_gpiRank, fid;
  gaspi_notification_t val = 1;
  // send to left, receive from the right:
  if (l_leftNeighborRank >= 0)
  {
    for(int j = 0; j < 3; j++)
    {
      for (int i = 0; i < l_nleftOutflowOffset; i++)
      {
        segment_id[i] = j;
        size[i] = sizeof(float);
      }
      val = 1;
      ASSERT(gaspi_write_list_notify(l_nleftOutflowOffset,
                                    segment_id,
                                    l_leftOutflowOffset,
                                    l_leftNeighborRank,
                                    segment_id,
                                    l_rightInflowOffset,
                                    size,
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
  if (l_rightNeighborRank >= 0)
  {
    for(int j = 0; j < 3; j++)
    {
      for (int i = 0; i < l_nrightOutflowOffset; i++)
      {
        segment_id[i] = j;
        size[i] = sizeof(float);
      }
      val = 1;
      ASSERT(gaspi_write_list_notify(l_nrightOutflowOffset,
                                    segment_id,
                                    l_rightOutflowOffset,
                                    l_rightNeighborRank,
                                    segment_id,
                                    l_leftInflowOffset,
                                    size,
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
  if(l_rightNeighborRank >= 0)
  {
    /*Wait for a notification from right neighbor for its write on my right inflow*/
    for(int j = 0; j < 3; j++)
    {
      ASSERT(gaspi_notify_waitsome(j, l_rightNeighborRank, 1, &fid, GASPI_BLOCK));
      val = 0;
      ASSERT (gaspi_notify_reset(j, fid, &val));
      //tools::Logger::logger.cout() << id << ": received segment " << j << std::endl;
    }
    //tools::Logger::logger.cout() << "received data from my right neighbor" << std::endl;
  }
  if(l_leftNeighborRank >= 0)
  {
    /*Wait for a notification from right neighbor for its write on my right inflow*/
    for(int j = 0; j < 3; j++)
    {
      ASSERT(gaspi_notify_waitsome(j, l_leftNeighborRank, 1, &fid, GASPI_BLOCK));
      val = 0;
      ASSERT (gaspi_notify_reset(j, fid, &val));
      //tools::Logger::logger.cout() << id << ": received segment " << j << std::endl;
    }
    //tools::Logger::logger.cout() << "received data from my left neighbor" << std::endl;
  }
}

void exchangeBottomTopGhostLayers( const int l_bottomNeighborRank,
                                   const int l_nbottomInflowOffset, 
                                   gaspi_offset_t *l_bottomInflowOffset,
                                   const int l_nbottomOutflowOffset,
                                   gaspi_offset_t *l_bottomOutflowOffset,
                                   const int l_topNeighborRank,
                                   const int l_ntopInflowOffset,
                                   gaspi_offset_t *l_topInflowOffset,
                                   int l_ntopOutflowOffset,
                                   gaspi_offset_t *l_topOutflowOffset,
                                   gaspi_segment_id_t *segment_id,
                                   gaspi_size_t *size,
                                   gaspi_rank_t l_gpiRank
                                  )
{
  // send to bottom, receive from the top:
  gaspi_notification_id_t id = l_gpiRank, fid;
  gaspi_notification_t val = 1;
  // send to left, receive from the right:
  if (l_bottomNeighborRank >= 0)
  {
    for(int j = 0; j < 3; j++)
    {
      for (int i = 0; i < l_nbottomOutflowOffset; i++)
      {
        segment_id[i] = j;
        size[i] = sizeof(float);
      }
      val = 1;
      ASSERT(gaspi_write_list_notify(l_nbottomOutflowOffset,
                                     segment_id,
                                     l_bottomOutflowOffset,
                                     l_bottomNeighborRank,
                                     segment_id,
                                     l_topInflowOffset,
                                     size,
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
  if (l_topNeighborRank >= 0)
  {
    for(int j = 0; j < 3; j++)
    {
      for (int i = 0; i < l_nbottomOutflowOffset; i++)
      {
        segment_id[i] = j;
        size[i] = sizeof(float);
      }
      val = 1;
      ASSERT(gaspi_write_list_notify(l_ntopOutflowOffset,
                                     segment_id,
                                     l_topOutflowOffset,
                                     l_topNeighborRank,
                                     segment_id,
                                     l_bottomInflowOffset,
                                     size,
                                     j,
                                     id,
                                     val,
                                     0,
                                     GASPI_BLOCK));
      ASSERT(gaspi_wait(0, GASPI_BLOCK));
    }
    //tools::Logger::logger.cout() << "sent data to my top neighbor" << std::endl;
  }
  if(l_topNeighborRank >= 0)
  {
    for(int j = 0; j < 3; j++)
    {
      ASSERT(gaspi_notify_waitsome(j, l_topNeighborRank, 1, &fid, GASPI_BLOCK));
      val = 0;
      ASSERT (gaspi_notify_reset(j, fid, &val));
    }
    //tools::Logger::logger.cout() << "received data from my top neighbor" << std::endl;
  }
  if(l_bottomNeighborRank >= 0)
  {
    for(int j = 0; j < 3; j++)
    {
      ASSERT(gaspi_notify_waitsome(j, l_bottomNeighborRank, 1, &fid, GASPI_BLOCK));
      val = 0;
      ASSERT (gaspi_notify_reset(j, fid, &val));
    }
    //tools::Logger::logger.cout() << "received data from my bottom neighbor" << std::endl;
  }
}
