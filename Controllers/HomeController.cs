﻿using Microsoft.AspNetCore.Mvc;
using System.Diagnostics;
using WebApplication3.Models;

namespace WebApplication3.Controllers
{
    public class HomeController : Controller
    {
        public IActionResult Index()
        {

            MLModel model = new MLModel();
            model.TrainAndEvaluate();

            ViewBag.Accuracy = model.Accuracy;
            return View(model);

        }
    }
}
